# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes and functions used for training (from-scratch and fine-tuning)."""
import fnmatch
import functools
import multiprocessing.pool
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.linen as nn
import flax.serialization
import flax.training.train_state
import flax.traverse_util
import jax
from jax import lax
from jax.experimental import pjit
from jax.interpreters import pxla
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
from vmoe import initialization
from vmoe import multihost_utils
from vmoe import partitioning
from vmoe import utils
from vmoe.data import input_pipeline
from vmoe.data import pjit_utils
from vmoe.evaluate import ensemble
from vmoe.evaluate import evaluator
from vmoe.evaluate import fewshot
from vmoe.nn import models
from vmoe.projects.adversarial_attacks import attacks as adversarial_attacks
from vmoe.train import cleanrl_ppo
from vmoe.train import hardware_context
from vmoe.train import optimizer
from vmoe.train import periodic_actions as train_periodic_actions
from vmoe.train import train_state as train_state_module
from vmoe.train import tree_summarizer

Array = jax.numpy.ndarray
ArraySpecDict = input_pipeline.ArraySpecDict
AsyncResult = multiprocessing.pool.AsyncResult
DatasetIterator = input_pipeline.DatasetIterator
Mesh = partitioning.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = partitioning.PartitionSpec
PRNGKey = jax.Array
PyTree = Any
ReportProgress = train_periodic_actions.ReportProgress
SingleProcessPeriodicAction = train_periodic_actions.SingleProcessPeriodicAction
ThreadPool = multiprocessing.pool.ThreadPool
TrainState = train_state_module.TrainState
TrainStateAxisResources = train_state_module.TrainStateAxisResources
TrainStepFn = train_state_module.TrainStepFn
TreeSummarizer = tree_summarizer.TreeSummarizer


def accumulate_gradients_and_metrics(grad_and_metrics_fn, microsteps: int):
  """Wraps a function that computes gradients and metrics to use microsteps.

  The new function will split the input `images` and `labels` into `microsteps`
  chunks, and will compute the gradients and metrics on each chunk, and average
  all gradients and metrics. A new PRNGKey is passed to each microstep.

  Args:
    grad_and_metrics_fn: A function that takes (params, images, labels, rngs)
      and returns gradients (same shape as params), and metrics (a dictionary
      with array leaves).
    microsteps: Positive integer, controlling the number of microsteps.

  Returns:
    A function with the same signature as grad_and_metric_fn.
  """
  if not microsteps or microsteps < 0:
    return grad_and_metrics_fn

  def new_grad_and_metrics_fn(params, images, labels, rngs):
    batch_size = images.shape[0]
    # TODO(jpuigcerver): Generalize for batch sizes that are not fully sharded.
    batch_size_per_device = batch_size // jax.device_count()
    assert batch_size_per_device % microsteps == 0
    # Note: we need this reshape because dynamic_index/dynamic_index_in_dim do
    # not work with strides, and we need to make sure that all acc steps cover
    # all devices.
    # TODO(jpuigcerver): Pass this PartitionSpec as an argument.
    pspec = jax.sharding.PartitionSpec(('expert', 'replica'))
    images = images.reshape((-1, microsteps) + images.shape[1:])
    labels = labels.reshape((-1, microsteps) + labels.shape[1:])
    images = lax.with_sharding_constraint(images, pspec)
    labels = lax.with_sharding_constraint(labels, pspec)

    def accum_fn(i, state):
      grad, rngs, metrics = state
      imgs = jax.lax.dynamic_index_in_dim(images, i, axis=1, keepdims=False)
      lbls = jax.lax.dynamic_index_in_dim(labels, i, axis=1, keepdims=False)
      new_grad, (next_rngs, new_metrics) = grad_and_metrics_fn(
          params, imgs, lbls, rngs)
      new_grad, new_metrics = jax.tree_util.tree_map(
          lambda x, y: x + y, (grad, metrics), (new_grad, new_metrics))
      return new_grad, next_rngs, new_metrics

    # Accumulate (sum) gradients and metrics through many microsteps.
    imgs = jax.lax.dynamic_index_in_dim(images, 0, axis=1, keepdims=False)
    lbls = jax.lax.dynamic_index_in_dim(labels, 0, axis=1, keepdims=False)
    grads, (next_rngs, metrics) = grad_and_metrics_fn(params, imgs, lbls, rngs)
    grads, next_rngs, metrics = jax.lax.fori_loop(
        1, microsteps, accum_fn, (grads, next_rngs, metrics))
    # Average gradients and metrics through all the microsteps.
    grads, metrics = jax.tree_util.tree_map(lambda x: x / microsteps,
                                            (grads, metrics))
    return grads, (next_rngs, metrics)

  return new_grad_and_metrics_fn


def create_checkpoint_manager(
    *,
    workdir: str,
    every_steps: int,
    keep_last: Optional[int] = None,
    keep_steps_multiple_of: Optional[int] = None,
    wait_seconds: int = 300,
) -> orbax.checkpoint.CheckpointManager:
  """Creates an Orbax checkpoint manager."""
  directory = os.path.join(workdir, 'ckpt')
  if jax.process_index() == 0 and not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  multihost_utils.sync_devices('create-ckpt-dir')
  ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
      save_interval_steps=every_steps,
      max_to_keep=keep_last,
      keep_period=keep_steps_multiple_of,
  )
  ckpt_manager = orbax.checkpoint.CheckpointManager(
      directory,
      {
          'state': orbax.checkpoint.AsyncCheckpointer(
              orbax.checkpoint.PyTreeCheckpointHandler(),
              timeout_secs=wait_seconds,
          ),
          'dataset_iterator': orbax.checkpoint.Checkpointer(
              orbax.checkpoint.JsonCheckpointHandler()
          ),
      },
      options=ckpt_options,
  )
  return ckpt_manager


def create_evaluation_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    datasets: Mapping[str, DatasetIterator],
    loss_fn: Callable[[Array, Array], Array],
    first_step: int,
    train_steps: int,
    extra_rng_keys: Sequence[str],
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    **kwargs,
) -> Tuple[evaluator.EvaluateMultipleDatasets, ml_collections.ConfigDict]:
  """Returns a hook to evaluate a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  apply_fn = create_flax_model(
      config=model_config.to_dict(), deterministic=True).apply
  on_steps = set(kwargs.pop('on_steps', []))
  # Always evaluate on the first and last step.
  on_steps.update([first_step, train_steps])
  periodic_action = evaluator.EvaluateMultipleDatasets(
      apply_fn=apply_fn,
      loss_fn=loss_fn,
      datasets=datasets,
      metric_writer=writer,
      rng_keys=extra_rng_keys,
      report_progress=progress_hook,
      report_progress_name='eval',
      on_steps=on_steps,
      **kwargs)
  return periodic_action, model_config


def create_fewshot_hook(
    *,
    base_model_config: ml_collections.ConfigDict,
    writer: metric_writers.MetricWriter,
    progress_hook: ReportProgress,
    first_step: int,
    train_steps: int,
    extra_rng_keys: Sequence[str],
    model_overrides: Optional[ml_collections.ConfigDict] = None,
    **kwargs) -> Tuple[Callable[..., Any], ml_collections.ConfigDict]:
  """Returns a hook to run fewshot evaluation of a model periodically."""
  model_config = override_base_config(base_model_config, model_overrides)
  # Few-shot eval requires additional mandatory parameters. If none of those is
  # given, we assume that no few-shot eval should be done.
  if not kwargs:
    return (lambda *args, **kw: None, model_config)
  apply_fn = create_flax_model(
      config=model_config.to_dict(), deterministic=True).apply
  on_steps = set(kwargs.pop('on_steps', []))
  # Always evaluate on the first and last step.
  on_steps.update([first_step, train_steps])
  periodic_action = fewshot.FewShotPeriodicAction(
      metric_writer=writer,
      apply_fn=apply_fn,
      rng_keys=extra_rng_keys,
      report_progress=progress_hook,
      report_progress_name='fewshot',
      on_steps=on_steps,
      **kwargs)
  return periodic_action, model_config


def create_flax_model(*, config: Dict[str, Any],
                      deterministic: bool) -> nn.Module:
  if 'name' not in config:
    raise KeyError('The model config must have a "name" field.')
  if isinstance(config, ml_collections.ConfigDict):
    config = config.to_dict()
  model_cls = config.pop('name')
  model_cls, args, kwargs = utils.parse_call(model_cls, models)
  return model_cls(*args, **kwargs, **config, deterministic=deterministic)


def create_profile_hook(*, workdir: str, **kwargs):
  all_processes = kwargs.pop('all_processes', False)
  if all_processes:
    return periodic_actions.ProfileAllHosts(logdir=workdir, **kwargs)
  else:
    return SingleProcessPeriodicAction(
        periodic_action=periodic_actions.Profile(logdir=workdir, **kwargs))


def create_progress_hook(*, writer: metric_writers.MetricWriter,
                         first_step: int, train_steps: int,
                         **kwargs) -> ReportProgress:
  on_steps = set(kwargs.pop('on_steps', []))
  # Always report progress on the first and last step.
  on_steps.update([first_step, train_steps])
  return ReportProgress(
      num_train_steps=train_steps, writer=writer, on_steps=on_steps, **kwargs)


def create_tree_summarizer(config) -> Optional[TreeSummarizer]:
  if not config:
    return None
  if isinstance(config, (list, tuple)):
    return TreeSummarizer(rules=config)
  else:
    return TreeSummarizer(**config)


def create_or_reuse_train_state(
    *,
    train_state: TrainState,
    initialize_fn: Callable[[], TrainState],
    mesh: Optional[Mesh] = None,
) -> TrainState:
  """Creates a TrainState object initialized from scratch, re-using some parts.

  Args:
    train_state: TrainState containing either jax.Array or jax.ShapeDtypeStruct
      leaves with sharding annotation.
    initialize_fn: Function used to create and initialize a train state from
      scratch.
    mesh: Logical mesh used by pjit. If None, uses the currently active mesh.

  Returns:
    A TrainState.
  """
  mesh = mesh or pxla.thread_resources.env.physical_mesh
  # Flatten input train state and keep the ShapeDtypeStruct for the arrays that
  # must be created from scratch.
  train_state_dict = flax.traverse_util.flatten_dict(
      flax.serialization.to_state_dict(train_state), sep='/',
      keep_empty_nodes=True)
  out_axis_resources = {
      k: v.sharding for k, v in train_state_dict.items()
      if isinstance(v, jax.ShapeDtypeStruct)
  }
  # Wrap initialize function to return only those arrays that are not present
  # yet in the input train state.
  def _initialize_fn():
    train_state = initialize_fn()
    train_state = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(train_state), sep='/',
        keep_empty_nodes=True)
    return {k: v for k, v in train_state.items() if k in out_axis_resources}
  # Run the wrapped initialize function, shard the outputs according to the
  # previously generated out_axis_resources dict.
  with mesh:
    new_train_state_dict = pjit.pjit(
        _initialize_fn, in_shardings=(), out_shardings=out_axis_resources
    )()
  # Replace the ShapeDtypeStruct objects from the input train state with the
  # newly generated arrays.
  for k, v in new_train_state_dict.items():
    train_state_dict[k] = v
  return flax.serialization.from_state_dict(
      train_state, flax.traverse_util.unflatten_dict(train_state_dict, sep='/'))


def get_dataset_iterator(
    dataset: DatasetIterator, prefetch_size: int, mesh: Mesh,
    last_seen_index: Optional[int] = None):
  """Creates a dataset iterator with device prefetching."""
  logging.warning("Your dataset iterator doesn't allow checkpointing!")
  del last_seen_index
  return pjit_utils.prefetch_to_device(dataset, size=prefetch_size, mesh=mesh)


def make_create_train_state_fn(
    *,
    model: nn.Module,
    optimizer_config: Dict[str, Any],
    input_shape_dtypes: Tuple[jax.ShapeDtypeStruct, ...],
    train_steps: int,
    seed: int = 0,
    extra_rng_keys: Tuple[str, ...] = (),
    init_routing_context: Optional[Mapping[str, Array]] = None,
) -> Callable[[], TrainState]:
  """Returns a function that creates and initializes a TrainState from scratch.

  Args:
    model: Linen module representing the model.
    optimizer_config: A ConfigDict with the optimizer configuration.
    input_shape_dtypes: ShapeDtypeStructs representing the inputs to the model.
    train_steps: Total number of training steps.
    seed: PRNG seed.
    extra_rng_keys: Sequence of RNG keys used by the model, in addition to
      "params".

  Returns:
    A function that creates a new TrainState when called.
  """
  # Optimizer is an optax.GradientTransform.
  tx = optimizer.create_optimizer(**optimizer_config, total_steps=train_steps)
  rng_keys = ('params',) + extra_rng_keys

  def initialize():
    rngs = utils.make_rngs(rng_keys, seed)
    inputs = tuple(jnp.zeros(x.shape, dtype=x.dtype)
                   for x in input_shape_dtypes)
    inputs = tuple(partitioning.with_sharding_constraint(x, s.sharding)
                   for x, s in zip(inputs, input_shape_dtypes))
    if init_routing_context is None:
      variables = model.init(rngs, *inputs)
    else:
      variables = model.init(
          rngs, *inputs, routing_context=init_routing_context)
    rngs.pop('params')  # This PRNGKey is not used anymore.
    return TrainState.create(
        apply_fn=model.apply, tx=tx, rngs=rngs, **variables)

  return initialize


def restore_or_create_train_state(
    *,
    ckpt_manager: orbax.checkpoint.CheckpointManager,
    initialize_fn: Callable[[], TrainState],
    axis_resources_regexes: partitioning.AxisResourcesRegexes,
    mesh: Optional[Mesh] = None,
    thread_pool: Optional[ThreadPool] = None,
    initialization_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[TrainState, Optional[int]]:
  """Restores a TrainState from the latest complete checkpoint or creates one.

  Args:
    ckpt_manager: Checkpoint manager.
    initialize_fn: Function used to create and initialize a train state from
      scratch.
    axis_resources_regexes: Regular expressions specifying how the TrainState
      is partitioned.
    mesh: Logical mesh used by pjit. If None, uses the currently active mesh.
    thread_pool: Thread pool used to restore checkpoints.
    initialization_kwargs: Optional dictionary containing the kwargs used to
      initialize the TrainState from an existing checkpoint.

  Returns:
    A TrainState and (optionally) the last_seen_index.
  """
  mesh = mesh or pxla.thread_resources.env.physical_mesh
  train_state_shape_dtype = jax.eval_shape(initialize_fn)
  train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=train_state_shape_dtype,
      axis_resources_regexes=axis_resources_regexes)
  train_state_axis_resources = jax.tree_util.tree_map(
      lambda spec: jax.sharding.NamedSharding(mesh, spec),
      train_state_axis_resources,
  )
  train_state = jax.tree_util.tree_map(
      lambda x, y: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=y),
      train_state_shape_dtype, train_state_axis_resources)

  if (step := ckpt_manager.latest_step()) is not None:
    logging.info('Continue training from checkpoint at step %d', step)
    def _array_restore_args_fn(x: jax.ShapeDtypeStruct):
      return orbax.checkpoint.ArrayRestoreArgs(
          dtype=x.dtype, sharding=x.sharding, global_shape=x.shape)
    restore_args = jax.tree_util.tree_map(_array_restore_args_fn, train_state)
    items = ckpt_manager.restore(
        step,
        args=orbax.checkpoint.args.Composite(
            state=orbax.checkpoint.args.PyTreeRestore(
                train_state, restore_args=restore_args
            ),
            dataset_iterator=orbax.checkpoint.args.JsonRestore(),
        ),
    )
    return items['state'], items['dataset_iterator']['last_seen_index']

  if initialization_kwargs:
    logging.info('Partially initializing the TrainState: %r',
                 initialization_kwargs)
    # Initialize TrainState from checkpoint. Arrays that are not initialized,
    # are ShapeDtypeStruct objects.
    train_state = initialize_train_state_from_checkpoint(
        train_state=train_state, mesh=mesh, thread_pool=thread_pool,
        **initialization_kwargs)
  if os.environ.get('VMOE_LOG_PARAMETER_OVERVIEW', '0') == '1':
    parameter_overview.log_parameter_overview(
        train_state_shape_dtype.params, include_stats=False,
        msg='Parameter overview:')
  return create_or_reuse_train_state(
      train_state=train_state, initialize_fn=initialize_fn, mesh=mesh), None


def get_loss_fn(name: str, **kwargs):
  """Returns the train/evaluation losses and the way to predict labels."""
  def default_sigmoid_xent(logits, labels, **kw):
    return jnp.sum(
        optax.sigmoid_binary_cross_entropy(logits, labels, **kw),
        axis=-1)

  default_label_pred_fn = lambda logits: jnp.argmax(logits, -1)
  loss_fns = {
      'softmax_xent': (
          optax.softmax_cross_entropy,
          optax.softmax_cross_entropy,
          default_label_pred_fn),
      'sigmoid_xent':
          (default_sigmoid_xent, default_sigmoid_xent, default_label_pred_fn),
      # In the case of the ensemble cross-entropy, the training loss and the
      # eval loss functions differ (see the discussion in Appendix D.1 of
      # https://arxiv.org/pdf/2110.03360.pdf).
      'ensemble_softmax_xent': (
          ensemble.ensemble_softmax_xent_train,
          ensemble.ensemble_softmax_xent_eval,
          ensemble.label_pred_ensemble_softmax),
      'ensemble_sigmoid_xent': (
          ensemble.ensemble_sigmoid_xent_train,
          ensemble.ensemble_sigmoid_xent_eval,
          ensemble.label_pred_ensemble_sigmoid),
  }
  if name in loss_fns:
    train_loss_fn, eval_loss_fn, pred_fn = loss_fns[name]
    train_loss_fn = functools.partial(train_loss_fn, **kwargs)
    eval_loss_fn = functools.partial(eval_loss_fn, **kwargs)
    if name.startswith('ensemble_'):
      # Pass ensemble_size to the pred_fn.
      pred_fn = functools.partial(pred_fn, **kwargs)
    return train_loss_fn, eval_loss_fn, pred_fn
  else:
    raise ValueError(f'Unknown loss: {name!r}')


def get_train_steps_and_epochs(
    *,
    train_batch_size: int,
    train_examples: int,
    train_steps: Optional[int] = None,
    train_epochs: Optional[Union[int, float]] = None,
) -> Tuple[int, float]:
  """Returns number of train steps and epochs."""
  if not train_steps and not train_epochs:
    raise ValueError('You must specify either `train_steps` or `train_epochs`')
  if train_steps is not None and train_epochs is not None:
    raise ValueError('You must specify either `train_steps` or `train_epochs`, '
                     f'but not both: train_steps = {train_steps!r}, '
                     f'train_epochs = {train_epochs!r}')
  if not train_steps:
    train_steps = int(np.ceil(train_epochs * train_examples / train_batch_size))
  train_epochs = train_steps * train_batch_size / train_examples
  return train_steps, train_epochs


def initialize_train_state_from_checkpoint(
    *,
    train_state: TrainState,
    name: str,
    mesh: Mesh,
    thread_pool: Optional[ThreadPool] = None,
    **kwargs) -> TrainState:
  """Initializes a TrainState from a pre-trained checkpoint.

  This is useful for fine-tuning a pre-trained model, where typically only
  (a subset of) the parameters of the TrainState are initialized, while the
  other from-scratch attributes of the TrainState are kept (e.g. optimizer
  state).

  Args:
    train_state: TrainState to initialize from the checkpoint. The given object
      must not be used after the call to this function. Use the returned one
      instead.
    name: Name of the method used to initialize the train state.
    mesh: Logical mesh used by pjit.
    thread_pool: Thread pool used to restore checkpoints.
    **kwargs: Additional arguments for the respective initialization function.

  Returns:
    A TrainState object.
  """
  if name == 'initialize_from_vmoe':
    return initialization.initialize_from_vmoe(
        target=train_state, mesh=mesh, thread_pool=thread_pool, **kwargs)
  elif name == 'initialize_from_vit':
    return initialization.initialize_from_vit(target=train_state, mesh=mesh,
                                              **kwargs)
  elif name == 'initialize_from_orbax':
    return initialization.initialize_from_orbax(target=train_state, mesh=mesh,
                                                **kwargs)
  else:
    raise ValueError(f'Unknown initialization method: {name!r}')


def make_train_cost_fn(compiled_fn) -> Callable[[int], Dict[str, float]]:
  """Returns a function that computes the total training cost at a given step."""
  flops_per_device, seconds_per_device = utils.get_flops_and_seconds_per_device(
      compiled_fn
  )

  def fn(step):
    output = {}
    if flops_per_device is not None:
      output['flops'] = flops_per_device * step * jax.device_count()
    if seconds_per_device is not None:
      output['device_seconds'] = seconds_per_device * step * jax.device_count()
    return output

  return fn


def mixup(
    rng: PRNGKey,
    tree: PyTree,
    *,
    concentration: float,
    shape: Tuple[int, ...] = (1, 2),
    roll_axis: int = 0,
    partition_spec: Optional[PartitionSpec] = None,
) -> Tuple[PRNGKey, PyTree]:
  """Performs mixup on each array of the given tree.

  For details on mixup, check "mixup: Beyond Empirical Risk Minimization"
  (https://arxiv.org/abs/1710.09412).

  The same mixing weights are used for all leaves of the tree, which is
  convenient to mix several arrays in the same way (e.g. images and labels).

  Args:
    rng: PRNGKey used to generate the mixup weights.
    tree: Tree with array leaves. The size of the first dimension of all trees
      must be equal, i.e. the batch_size.
    concentration: Dirichlet concentration parameter, used to sample the
      mixing weights. Must be a positive float.
    shape: Shape of the samples from the Dirichlet distribution.
    roll_axis: Axis to roll to mix examples. This is typically the 'batch'
      axis from your input arrays. It must be [0, len(shape) - 1).
    partition_spec: Optional PartitionSpec used to annotate the sampled values
      from the Dirichlet distribution.

  Returns:
    A tree with the mixed arrays.
  """
  arrays, treedef = jax.tree_util.tree_flatten(tree)
  if len(shape) < 2:
    raise ValueError(f"Mixup 'shape' has length {len(shape)}, but it must have "
                     'length >= 2.')
  # Check that all arrays have the same batch size.
  batch_ndim = len(shape) - 1
  for i, x in enumerate(arrays):
    try:
      _ = jnp.broadcast_shapes(x.shape[:batch_ndim], shape[:-1])
    except ValueError:
      raise ValueError(f'Mixup with inconsistent shapes. The shape of the '
                       f'{i}-th array is {x.shape}, but the first {batch_ndim} '
                       f'dims must be broadcastable to {shape[:-1]}') from None
    if x.shape[:batch_ndim] != arrays[0].shape[:batch_ndim]:
      raise ValueError(f'Mixup with inconsistent shapes. The shape of the '
                       f'{i}-th array is {x.shape}, but the first {batch_ndim} '
                       f'dims must be equal to {arrays[0].shape[:batch_ndim]}')
  if not 0 <= roll_axis < batch_ndim:
    raise ValueError("Mixup 'roll_axis' must be an integer in "
                     f'[0, {batch_ndim}), but got roll_axis = {roll_axis}')
  # Check mixup parameters.
  if concentration <= 0.:
    raise ValueError(f"Mixup 'concentration' must be greater than 0, but got "
                     f'concentration = {concentration}')
  # Generate alphas (weights) for the mixup.
  concentration = jnp.full(shape, concentration)
  concentration = partitioning.with_sharding_constraint(
      concentration, partition_spec)
  alpha = jax.random.dirichlet(rng, concentration)
  alpha = partitioning.with_sharding_constraint(alpha, partition_spec)
  # Put the largest weight of each example in the first position of alpha.
  # This avoids destroying examples due to permuations of the weights.
  # If one samples an alpha for every example, then chances are that many of
  # the original examples are lost (with mixup_size = 2).
  # Suppose that we have concentration approaching 0, and mixup_size = 2.
  # Then each alpha (Dirichlet sample) will be either (1, 0) or (0, 1).
  # If the batch is [1, 2], with alpha [(1, 0), (0, 1)], then the output will
  # be [1, 1]. If the sampled alpha is [(0, 1), (1, 0)], the output is [2, 2].
  alpha = -jnp.sort(-alpha, axis=-1)
  # Reshape alphas to (mixup_size, batch_size, ...) for convenience.
  alpha = jnp.moveaxis(alpha, -1, 0)
  # Mix each sample with the next `mixup_size` samples within the same group.
  def mix(x):
    a = alpha.reshape(alpha.shape + (1,) * (x.ndim - batch_ndim))
    return sum(a[i] * jnp.roll(x, -i, axis=roll_axis) for i in range(shape[-1]))
  arrays = list(map(mix, arrays))
  return jax.tree.unflatten(treedef, arrays)


def override_base_config(
    base: ml_collections.ConfigDict,
    override: Optional[ml_collections.ConfigDict],
) -> ml_collections.ConfigDict:
  output = base.copy_and_resolve_references()
  with output.unlocked():
    if override:
      output.update(override)
  return output


def train_step(
    state: TrainState,
    images: Array,
    labels: Array,
    routing_context: Optional[Mapping[str, Array]] = None,
    *,
    loss_fn: Callable[[Array, Array], Array],
    microsteps: Optional[int] = None,
    summarizer: Optional[TreeSummarizer] = None,
    rl_loss_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[TrainState, Mapping[str, Any]]:
  """Performs one update step of the given TrainState object ."""

  @functools.partial(jax.grad, has_aux=True)
  def compute_grads_and_metrics(params, images, labels, rngs):
    rngs, next_rngs = utils.tree_rngs_split(rngs)
    logits, metrics = state.apply_fn(
        {'params': params}, images, routing_context=routing_context, rngs=rngs)
    metrics = dict(**metrics)
    if routing_context:
      metrics.update({
          f"routing_context/input/{name}": jnp.asarray(value, dtype=jnp.float32)
          for name, value in routing_context.items()
      })
    metrics['main_loss'] = jnp.mean(loss_fn(logits, labels))
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)

    def _mean_metric_by_suffix(suffix):
      flat_metrics = flax.traverse_util.flatten_dict(metrics, sep='/')
      values = [
          jnp.asarray(value) for name, value in flat_metrics.items()
          if name == suffix or name.endswith('/' + suffix)
      ]
      if not values:
        return None
      return jnp.mean(jnp.stack(values))

    rl_loss_kwargs = dict(**(rl_loss_config or {}))
    miss_weight = float(rl_loss_kwargs.get('miss_weight', 1.0))
    usage_entropy_weight = float(
        rl_loss_kwargs.get('usage_entropy_weight', 0.01))
    selection_budget_weight = float(
        rl_loss_kwargs.get('selection_budget_weight', 0.0))
    miss_loss = _mean_metric_by_suffix(
        'routing/variable_k/importance_weighted_miss')
    if miss_loss is not None:
      usage_entropy = _mean_metric_by_suffix('routing/variable_k/usage_entropy')
      if usage_entropy is None:
        usage_entropy = jnp.asarray(0.0, dtype=jnp.float32)
      selection_budget_loss = _mean_metric_by_suffix(
          'routing/variable_k/selection_budget_loss')
      if selection_budget_loss is None:
        selection_budget_loss = _mean_metric_by_suffix(
            'routing/variable_k/avg_selected_k_ratio')
      if selection_budget_loss is None:
        selection_budget_loss = jnp.asarray(0.0, dtype=jnp.float32)
      metrics['routing/variable_k/importance_weighted_miss'] = miss_loss
      metrics['routing/variable_k/usage_entropy'] = usage_entropy
      metrics['routing/variable_k/selection_budget_loss'] = selection_budget_loss
      rl_loss = (
          miss_weight * miss_loss
          - usage_entropy_weight * usage_entropy
          + selection_budget_weight * selection_budget_loss)
      metrics['rl_loss/miss_component'] = miss_weight * miss_loss
      metrics['rl_loss/usage_entropy_component'] = (
          -usage_entropy_weight * usage_entropy)
      metrics['rl_loss/selection_budget_component'] = (
          selection_budget_weight * selection_budget_loss)
    else:
      rl_loss = jnp.asarray(0.0, dtype=jnp.float32)

    metrics['rl_loss'] = rl_loss

    total_loss = (
        metrics['main_loss']
        + metrics.get('auxiliary_loss', 0.0)
        + metrics['rl_loss']
    )

    metrics['total_loss'] = total_loss
    
    return total_loss, (next_rngs, metrics)

  compute_grads_and_metrics = accumulate_gradients_and_metrics(
      compute_grads_and_metrics, microsteps)
  grads, (next_rngs, metrics) = compute_grads_and_metrics(
      state.params, images, labels, state.rngs)
  state, global_norms = state.apply_gradients_and_compute_global_norms(
      grads, rngs=next_rngs)
  metrics.update({f'global_norm/{k}': v for k, v in global_norms.items()})

  if summarizer:
    # Summarize arrays in the gradients tree or the train state.
    state_flat = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(state), sep='/')
    state_flat['params_grads'] = flax.traverse_util.flatten_dict(grads, sep='/')
    metrics.update(summarizer(state_flat))

  return state, metrics


def _metric_values_by_suffix(metrics: Mapping[str, Any], suffix: str):
  flat_metrics = flax.traverse_util.flatten_dict(metrics, sep='/')
  return [
      value for name, value in flat_metrics.items()
      if name == suffix or name.endswith('/' + suffix)
  ]


def _stack_metric_by_suffix(
    metrics: Mapping[str, Any], suffix: str, dtype: Optional[Any] = None
) -> Array:
  values = _metric_values_by_suffix(metrics, suffix)
  if not values:
    raise KeyError(f'No metric ending with {suffix!r} was found.')
  arrays = [jnp.asarray(value, dtype=dtype) for value in values]
  return jnp.stack(arrays, axis=0)


def _mean_metric_by_suffix(
    metrics: Mapping[str, Any], suffix: str, default: Optional[Array] = None
) -> Optional[Array]:
  values = _metric_values_by_suffix(metrics, suffix)
  if not values:
    return default
  return jnp.mean(jnp.stack([jnp.asarray(value) for value in values]))


def _mean_metrics_for_logging(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
  return jax.tree_util.tree_map(
      lambda x: jnp.mean(jnp.asarray(x, dtype=jnp.float32)), metrics)


_DEFAULT_COMPACT_TRAIN_ALLOWLIST = (
    'main_loss',
    'rl_loss',
    'global_norm/grads',
    'global_norm/updates',
    'ppo/policy_loss',
    'ppo/value_loss',
    'ppo/entropy_loss',
    'ppo/approx_kl',
    'ppo/clipfrac',
    'ppo/ratio_mean',
    'ppo/return_mean',
    'ppo/value_mean',
    'rl_loss/miss_reward_component',
    'rl_loss/usage_entropy_reward_component',
    'rl_loss/latency_reward_component',
    'routing/variable_k/latency_reward',
    'routing/variable_k/latency_baseline_secs',
    'routing/variable_k/avg_selected_k',
    'routing/variable_k/avg_selected_k_ratio',
    'routing/variable_k/selected_q_mass',
    'routing/variable_k/accepted_q_mass',
    'routing/variable_k/importance_weighted_miss',
    'routing/variable_k/accepted_usage_entropy',
    'routing/variable_k/max_expert_usage_ratio',
    'routing/variable_k/policy_entropy',
    'routing/variable_k/selected_k_ratio/k_*',
    'routing/variable_k/block_*/avg_selected_k',
    'routing/variable_k/block_*/avg_selected_k_ratio',
    'routing/variable_k/block_*/selected_q_mass',
    'routing/variable_k/block_*/accepted_q_mass',
    'routing/variable_k/block_*/importance_weighted_miss',
    'routing/variable_k/block_*/accepted_usage_entropy',
    'routing/variable_k/block_*/max_expert_usage_ratio',
    'routing/variable_k/block_*/policy_entropy',
    'routing/variable_k/block_*/selected_k_ratio/k_*',
)


def _add_variable_k_block_metric_aliases(
    flat_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
  """Adds block_N aliases for nested encoderblock_N variable-K metrics."""
  metrics_with_aliases = dict(flat_metrics)
  metric_prefix = 'routing/variable_k/'
  for name, value in flat_metrics.items():
    parts = name.split('/', 1)
    if len(parts) != 2 or not parts[0].startswith('encoderblock_'):
      continue
    block_name, suffix_path = parts
    if not suffix_path.startswith(metric_prefix):
      continue
    block_id = block_name.removeprefix('encoderblock_')
    suffix = suffix_path[len(metric_prefix):]
    metrics_with_aliases[
        f'routing/variable_k/block_{block_id}/{suffix}'] = value
  return metrics_with_aliases


def _matches_metric_patterns(name: str, patterns: Sequence[str]) -> bool:
  return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _filter_train_metrics_for_logging(
    metrics: Mapping[str, Any],
    metric_logging_config: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
  if not metric_logging_config or not metric_logging_config.get(
      'compact_train_metrics', False):
    return metrics
  flat_metrics = flax.traverse_util.flatten_dict(metrics, sep='/')
  flat_metrics = _add_variable_k_block_metric_aliases(flat_metrics)
  allowlist = tuple(metric_logging_config.get(
      'train_allowlist', _DEFAULT_COMPACT_TRAIN_ALLOWLIST))
  return {
      name: value for name, value in flat_metrics.items()
      if _matches_metric_patterns(name, allowlist)
  }

def variable_k_ppo_collect_step(
    state: TrainState,
    images: Array,
    labels: Array,
    routing_context: Optional[Mapping[str, Array]] = None,
    *,
    loss_fn: Callable[[Array, Array], Array],
) -> Tuple[TrainState, Mapping[str, Any], Mapping[str, Array]]:
  """Collects one contextual-bandit PPO transition for variable-K routing."""
  rngs, next_rngs = utils.tree_rngs_split(state.rngs)
  logits, metrics = state.apply_fn(
      {'params': state.params}, images, routing_context=routing_context,
      rngs=rngs)
  metrics = dict(**metrics)
  if routing_context:
    metrics.update({
        f"routing_context/input/{name}": jnp.asarray(value, dtype=jnp.float32)
        for name, value in routing_context.items()
        if name != 'variable_k/action_ids'
    })
  metrics['main_loss'] = jnp.mean(loss_fn(logits, labels))
  auxiliary_loss = jnp.asarray(metrics.get('auxiliary_loss', 0.0),
                               dtype=jnp.float32)
  usage_entropy = _mean_metric_by_suffix(
      metrics, 'routing/variable_k/accepted_usage_entropy')
  if usage_entropy is None:
    usage_entropy = _mean_metric_by_suffix(
        metrics, 'routing/variable_k/usage_entropy',
        jnp.asarray(0.0, dtype=jnp.float32))
  ppo_batch = {
      'action_ids': _stack_metric_by_suffix(
          metrics, 'routing/variable_k/ppo_action_id', dtype=jnp.int32),
      'old_log_prob': _stack_metric_by_suffix(
          metrics, 'routing/variable_k/ppo_action_log_prob', dtype=jnp.float32),
      'old_value': _stack_metric_by_suffix(
          metrics, 'routing/variable_k/ppo_value', dtype=jnp.float32),
      'accepted_q_mass': _stack_metric_by_suffix(
          metrics, 'routing/variable_k/accepted_q_mass_per_token',
          dtype=jnp.float32),
      'usage_entropy': jnp.asarray(usage_entropy, dtype=jnp.float32),
  }
  logging_metrics = dict(_mean_metrics_for_logging(metrics))
  logging_metrics['total_loss'] = logging_metrics['main_loss'] + jnp.mean(
      auxiliary_loss)
  return state.replace(rngs=next_rngs), logging_metrics, ppo_batch


def variable_k_ppo_update_step(
    state: TrainState,
    images: Array,
    labels: Array,
    ppo_batch: Mapping[str, Array],
    latency_reward: Array,
    routing_context: Optional[Mapping[str, Array]] = None,
    *,
    loss_fn: Callable[[Array, Array], Array],
    ppo_config: Optional[Mapping[str, Any]] = None,
    summarizer: Optional[TreeSummarizer] = None,
) -> Tuple[TrainState, Mapping[str, Any]]:
  """Updates variable-K adapter params with CleanRL-style clipped PPO."""
  ppo_kwargs = dict(**(ppo_config or {}))
  miss_reward_weight = float(ppo_kwargs.get('miss_reward_weight', 1.0))
  usage_entropy_weight = float(ppo_kwargs.get('usage_entropy_weight', 0.01))
  latency_weight = float(ppo_kwargs.get('latency_weight', 0.01))
  clip_coef = float(ppo_kwargs.get('clip_coef', 0.2))
  vf_coef = float(ppo_kwargs.get('vf_coef', 0.5))
  ent_coef = float(ppo_kwargs.get('ent_coef', 0.01))
  normalize_advantage = bool(ppo_kwargs.get('advantage_normalize', True))
  update_epochs = int(ppo_kwargs.get('update_epochs', 1))
  num_minibatches = int(ppo_kwargs.get('num_minibatches', 1))
  if update_epochs < 1:
    raise ValueError(f'update_epochs must be >= 1, got {update_epochs}.')
  if num_minibatches != 1:
    raise ValueError('Only num_minibatches=1 is supported for routing PPO V1.')

  def compute_grads_and_metrics(params, rngs):

    @functools.partial(jax.grad, has_aux=True)
    def compute_loss(params):
      step_rngs, next_rngs = utils.tree_rngs_split(rngs)
      update_routing_context = dict(routing_context or {})
      update_routing_context['variable_k/action_ids'] = ppo_batch['action_ids']
      logits, metrics = state.apply_fn(
          {'params': params}, images, routing_context=update_routing_context,
          rngs=step_rngs)
      metrics = dict(**metrics)
      metrics['main_loss'] = jnp.mean(loss_fn(logits, labels))

      new_log_prob = _stack_metric_by_suffix(
          metrics, 'routing/variable_k/ppo_action_log_prob', dtype=jnp.float32)
      new_values = _stack_metric_by_suffix(
          metrics, 'routing/variable_k/ppo_value', dtype=jnp.float32)
      entropy = _stack_metric_by_suffix(
          metrics, 'routing/variable_k/ppo_policy_entropy', dtype=jnp.float32)

      latency_reward_f = jnp.asarray(latency_reward, dtype=jnp.float32)
      returns = (
          miss_reward_weight * ppo_batch['accepted_q_mass']
          + usage_entropy_weight * ppo_batch['usage_entropy']
          + latency_weight * latency_reward_f)
      old_values = ppo_batch['old_value']
      advantages = returns - old_values
      ppo_loss, ppo_metrics = cleanrl_ppo.clipped_ppo_loss(
          new_log_prob=new_log_prob,
          old_log_prob=ppo_batch['old_log_prob'],
          advantages=advantages,
          new_values=new_values,
          old_values=old_values,
          returns=returns,
          entropy=entropy,
          clip_coef=clip_coef,
          vf_coef=vf_coef,
          ent_coef=ent_coef,
          normalize_advantage=normalize_advantage)

      logging_metrics = dict(_mean_metrics_for_logging(metrics))
      logging_metrics.update({
          'rl_loss': ppo_loss,
          'ppo/policy_loss': ppo_metrics['ppo/policy_loss'],
          'ppo/value_loss': ppo_metrics['ppo/value_loss'],
          'ppo/entropy_loss': ppo_metrics['ppo/entropy_loss'],
          'ppo/approx_kl': ppo_metrics['ppo/approx_kl'],
          'ppo/old_approx_kl': ppo_metrics['ppo/old_approx_kl'],
          'ppo/clipfrac': ppo_metrics['ppo/clipfrac'],
          'ppo/ratio_mean': ppo_metrics['ppo/ratio_mean'],
          'ppo/advantage_mean': ppo_metrics['ppo/advantage_mean'],
          'ppo/return_mean': ppo_metrics['ppo/return_mean'],
          'ppo/value_mean': ppo_metrics['ppo/value_mean'],
          'routing/variable_k/ppo_approx_kl': ppo_metrics['ppo/approx_kl'],
          'routing/variable_k/ppo_clipfrac': ppo_metrics['ppo/clipfrac'],
          'routing/variable_k/latency_reward': latency_reward_f,
          'rl_loss/miss_reward_component': (
              miss_reward_weight * jnp.mean(ppo_batch['accepted_q_mass'])),
          'rl_loss/usage_entropy_reward_component': (
              usage_entropy_weight * ppo_batch['usage_entropy']),
          'rl_loss/latency_reward_component': latency_weight * latency_reward_f,
      })
      compact_metric_suffixes = (
          'routing/variable_k/avg_selected_k',
          'routing/variable_k/avg_selected_k_ratio',
          'routing/variable_k/selected_q_mass',
          'routing/variable_k/accepted_q_mass',
          'routing/variable_k/importance_weighted_miss',
          'routing/variable_k/usage_entropy',
          'routing/variable_k/accepted_usage_entropy',
          'routing/variable_k/max_expert_usage_ratio',
          'routing/variable_k/policy_entropy',
          'routing/variable_k/policy_log_prob',
          'routing/variable_k/value_mean',
      )
      for suffix in compact_metric_suffixes:
        value = _mean_metric_by_suffix(metrics, suffix)
        if value is not None:
          logging_metrics[suffix] = value
      for k in range(1, 17):
        suffix = f'routing/variable_k/selected_k_ratio/k_{k}'
        value = _mean_metric_by_suffix(metrics, suffix)
        if value is not None:
          logging_metrics[suffix] = value
      logging_metrics['total_loss'] = ppo_loss
      return ppo_loss, (next_rngs, logging_metrics)

    return compute_loss(params)

  original_step = state.step
  metrics = {}
  global_norms = {}
  for _ in range(update_epochs):
    grads, (next_rngs, metrics) = compute_grads_and_metrics(
        state.params, state.rngs)
    state, global_norms = state.apply_gradients_and_compute_global_norms(
        grads, rngs=next_rngs)
  state = state.replace(step=original_step + 1)
  metrics.update({f'global_norm/{k}': v for k, v in global_norms.items()})

  if summarizer:
    state_flat = flax.traverse_util.flatten_dict(
        flax.serialization.to_state_dict(state), sep='/')
    state_flat['params_grads'] = flax.traverse_util.flatten_dict(grads, sep='/')
    metrics.update(summarizer(state_flat))

  return state, metrics


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str,
                       mesh: Mesh, writer: metric_writers.MetricWriter):
  """Trains a model and evaluates it periodically."""
  datasets = input_pipeline.get_datasets(config.dataset)
  if 'train' not in datasets:
    raise KeyError(f'You must have a "train" variant of the dataset. '
                   f'Available variants are {sorted(datasets.keys())!r}')
  train_examples = input_pipeline.get_data_num_examples(config.dataset.train)
  train_batch_size = config.dataset.train.batch_size
  train_steps, train_epochs = get_train_steps_and_epochs(
      train_steps=config.get('train_steps'),
      train_epochs=config.get('train_epochs'),
      train_batch_size=train_batch_size,
      train_examples=train_examples)
  logging.info(
      'Training for %d steps (%g epochs) over %d examples, with a '
      'batch size of %d', train_steps, train_epochs, train_examples,
      train_batch_size)

  # Get the global shape of the image array.
  datataset_element_shape_dtype = pjit_utils.get_dataset_shape_dtype_struct(
      datasets['train'])

  ckpt_manager = create_checkpoint_manager(
      workdir=workdir, **config.get('save_checkpoint', {}))
  use_routing_context = bool(config.get('routing_context', {}).get(
      'enabled', False))

  def _make_routing_context(step: int):
    if not use_routing_context:
      return None
    return hardware_context.make_routing_context(
        config, step=step, train_steps=train_steps, batch_size=train_batch_size)

  train_state_initialize_fn = make_create_train_state_fn(
      model=create_flax_model(config=config.model, deterministic=False),
      optimizer_config=config.optimizer,
      input_shape_dtypes=(datataset_element_shape_dtype['image'],),
      train_steps=train_steps,
      extra_rng_keys=tuple(config.get('extra_rng_keys', [])),
      init_routing_context=_make_routing_context(0),
      seed=config.get('seed', 0))
  train_state, last_seen_index = restore_or_create_train_state(
      ckpt_manager=ckpt_manager,
      initialize_fn=train_state_initialize_fn,
      axis_resources_regexes=config.params_axis_resources,
      thread_pool=ThreadPool(),
      initialization_kwargs=config.get('initialization'))
  init_step = int(train_state.step)
  logging.info('Initial step = %d', init_step)
  tr_iter = get_dataset_iterator(
      dataset=datasets['train'],
      prefetch_size=config.dataset.train.get('prefetch_device', 1),
      mesh=mesh,
      last_seen_index=last_seen_index)
  train_loss_fn, eval_loss_fn, label_pred_fn = get_loss_fn(**config.loss)
  summarizer = create_tree_summarizer(config.get('summarize_arrays'))
  ppo_config = config.get('ppo', {})
  ppo_enabled = bool(ppo_config.get('enabled', False))
  latency_weight = float(ppo_config.get('latency_weight', 0.0))
  latency_baseline_secs = float(ppo_config.get('latency_baseline_secs', 0.0))
  if ppo_enabled and latency_weight > 0.0 and latency_baseline_secs <= 0.0:
    raise ValueError(
        'config.ppo.latency_baseline_secs must be > 0 when '
        'config.ppo.latency_weight > 0.')

  if ppo_enabled:
    if config.get('adversarial', {}):
      raise ValueError('PPO variable-k training does not support adversarial.')
    if config.get('mixup', {}):
      raise ValueError('PPO variable-k training does not support mixup.')
    if config.get('microsteps'):
      raise ValueError('PPO variable-k training does not support microsteps.')
    train_step_fn = functools.partial(
        variable_k_ppo_collect_step,
        loss_fn=train_loss_fn)
    ppo_update_step_fn = functools.partial(
        variable_k_ppo_update_step,
        loss_fn=train_loss_fn,
        ppo_config=ppo_config,
        summarizer=summarizer)
    train_step_pjit = pjit.pjit(
        fun=train_step_fn,
        out_shardings=(
            jax.tree_util.tree_map(lambda x: x.sharding, train_state),
            None,
            None,
        ),
        donate_argnums=(0,),
    )
    ppo_update_step_pjit = pjit.pjit(
        fun=ppo_update_step_fn,
        out_shardings=(
            jax.tree_util.tree_map(lambda x: x.sharding, train_state),
            None,
        ),
        donate_argnums=(0,),
    )
  else:
    train_step_fn = functools.partial(
        train_step,
        loss_fn=train_loss_fn,
        microsteps=config.get('microsteps'),
        summarizer=summarizer,
        rl_loss_config=config.get('rl_loss', {}))
    if config.get('adversarial', {}):
      adversarial_config = config.adversarial.to_dict()
      train_step_fn = wrap_train_step_with_adversarial_attack(
          train_step_fn, train_loss_fn, **adversarial_config)
    # If mixup options are defined, wrap the train_step_fn with mixup.
    if config.get('mixup', {}):
      mixup_config = config.mixup.to_dict()
      train_step_fn = wrap_train_step_with_mixup(
          train_step_fn,
          partition_spec=jax.sharding.PartitionSpec(mesh.axis_names,),
          **mixup_config)

    train_step_pjit = pjit.pjit(
        fun=train_step_fn,
        out_shardings=(
            jax.tree_util.tree_map(lambda x: x.sharding, train_state),
            None,
        ),
        donate_argnums=(0, 1, 2),
    )

  # Setup hooks.
  profile_hook = create_profile_hook(
      workdir=workdir, **config.get('profile', {}))
  progress_hook = create_progress_hook(
      writer=writer, first_step=init_step + 1, train_steps=train_steps,
      **config.get('report_progress', {}))
  evaluation_hook, config_model_eval = create_evaluation_hook(
      base_model_config=config.model.copy_and_resolve_references(),
      writer=writer,
      progress_hook=progress_hook,
      datasets={name: ds for name, ds in datasets.items() if name != 'train'},
      loss_fn=eval_loss_fn,
      label_pred_fn=label_pred_fn,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('evaluate', {}))
  fewshot_hook, _ = create_fewshot_hook(
      base_model_config=config_model_eval,
      writer=writer,
      progress_hook=progress_hook,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('fewshot', {}))
  # Run checkpoint hook just before starting the loop. This will save the train
  # state at initialization.
  def _save_checkpoint(step, ts, it, force=False):
    if config.get('disable_checkpoint_save', False):
      return
    last_seen_index = step * train_batch_size
    with progress_hook.timed('ckpt', wait_jax_async_dispatch=False):
      ckpt_manager.save(
          step,
          items={
              'state': ts,
              'dataset_iterator': {'last_seen_index': last_seen_index},
          },
          force=force)
  if init_step == 0 and not tf.io.gfile.exists(os.path.join(workdir, 'ckpt/0')):
    multihost_utils.sync_devices('training:ckpt-first')
    _save_checkpoint(init_step, train_state, tr_iter, force=True)
  # Explicitly compile train_step here.
  t0 = time.time()
  compile_args = [
      train_state,
      datataset_element_shape_dtype['image'],
      datataset_element_shape_dtype['labels'],
  ]
  if use_routing_context:
    compile_args.append(_make_routing_context(init_step + 1))
  train_step_pjit = train_step_pjit.lower(*compile_args).compile()
  t1 = time.time()
  metric_logging_config = config.get('metric_logging', {})
  log_train_cost = bool(metric_logging_config.get('log_train_cost', True))
  if log_train_cost:
    writer.write_scalars(init_step + 1, {'train/compile_secs': t1 - t0})
    train_step_flops_per_device, train_step_seconds_per_device = (
        utils.get_flops_and_seconds_per_device(train_step_pjit))
    if train_step_flops_per_device:
      writer.write_scalars(
          init_step + 1,
          {'train/step_flops_per_device': train_step_flops_per_device})
    if train_step_seconds_per_device:
      writer.write_scalars(
          init_step + 1,
          {'train/step_seconds_per_device': train_step_seconds_per_device})
    train_cost_fn = make_train_cost_fn(train_step_pjit)
  else:
    train_cost_fn = lambda step: {}
  for step, batch in zip(range(init_step + 1, train_steps + 1), tr_iter):
    profile_hook(step)
    if ppo_enabled:
      collect_t0 = time.time()
      with jax.profiler.StepTraceAnnotation('train_collect', step_num=step):
        if use_routing_context:
          train_state, collect_metrics, ppo_batch = train_step_pjit(
              train_state, batch['image'], batch['labels'],
              _make_routing_context(step))
        else:
          train_state, collect_metrics, ppo_batch = train_step_pjit(
              train_state, batch['image'], batch['labels'])
      jax.block_until_ready(collect_metrics['total_loss'])
      wall_time_per_step_secs = time.time() - collect_t0
      if latency_baseline_secs > 0.0:
        latency_reward = -(
            wall_time_per_step_secs / latency_baseline_secs - 1.0)
      else:
        latency_reward = 0.0

      update_t0 = time.time()
      with jax.profiler.StepTraceAnnotation('train_ppo_update', step_num=step):
        if use_routing_context:
          train_state, metrics = ppo_update_step_pjit(
              train_state, batch['image'], batch['labels'], ppo_batch,
              jnp.asarray(latency_reward, dtype=jnp.float32),
              _make_routing_context(step))
        else:
          train_state, metrics = ppo_update_step_pjit(
              train_state, batch['image'], batch['labels'], ppo_batch,
              jnp.asarray(latency_reward, dtype=jnp.float32))
      jax.block_until_ready(metrics['total_loss'])
      metrics = dict(metrics)
      metrics['wall_time_per_step_secs'] = wall_time_per_step_secs
      metrics['ppo_update_time_secs'] = time.time() - update_t0
      metrics['routing/variable_k/latency_reward'] = latency_reward
      metrics['routing/variable_k/latency_baseline_secs'] = (
          latency_baseline_secs)
    else:
      step_t0 = time.time()
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        if use_routing_context:
          train_state, metrics = train_step_pjit(
              train_state, batch['image'], batch['labels'],
              _make_routing_context(step))
        else:
          train_state, metrics = train_step_pjit(
              train_state, batch['image'], batch['labels'])
      jax.block_until_ready(metrics['total_loss'])
      metrics = dict(metrics)
      metrics['wall_time_per_step_secs'] = time.time() - step_t0
    metrics_for_logging = _filter_train_metrics_for_logging(
        metrics, metric_logging_config)
    progress_hook(step, scalar_metrics={
        f'train/{k}': v for k, v in metrics_for_logging.items()
    })
    _save_checkpoint(step, train_state, tr_iter)
    evaluation_hook(step, params=train_state.params, **train_cost_fn(step))
    fewshot_hook(step, variables={'params': train_state.params},
                 **train_cost_fn(step))
  ckpt_manager.wait_until_finished()
  if not tf.io.gfile.exists(os.path.join(workdir, f'ckpt/{train_steps}')):
    multihost_utils.sync_devices('training:ckpt-last')
    _save_checkpoint(train_steps, train_state, tr_iter, force=True)
    ckpt_manager.wait_until_finished()
  multihost_utils.sync_devices('training:completed')
  logging.info('Training completed.')


def wrap_train_step_with_adversarial_attack(
    train_step_fn: TrainStepFn,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    attack_auxiliary_loss: bool,
    max_epsilon: float,
    num_updates: int,
) -> TrainStepFn:
  """Wraps a train_step_fn to perform PGD adversarial training.

  Arguments:
    train_step_fn: Training step function to wrap.
    loss_fn: Loss function to attack.
    attack_auxiliary_loss: Whether to attack the MoE auxiliary loss or not.
    max_epsilon: Maximum change for each pixel.
    num_updates: Number of PGD updates to perform in the attack.

  Returns:
    A new train_step_fn.
  """
  if num_updates <= 0 or max_epsilon <= 0.0:
    return train_step_fn

  def loss_with_auxiliary_attack_fn(logits, labels, metrics):
    loss = loss_fn(logits, labels)
    if attack_auxiliary_loss:
      loss = loss + jnp.mean(metrics['auxiliary_loss'])
    return loss

  def new_train_step_fn(
      state: TrainState,
      images: jnp.ndarray,
      labels: jnp.ndarray,
      *args,
      **kwargs,
  ) -> Tuple[TrainState, Mapping[str, Any]]:
    def apply_fn(x, **kwargs):
      return state.apply_fn({'params': state.params}, x, **kwargs)
    images, rngs = adversarial_attacks.stateless_attack_pgd(
        images=images, labels=labels, rngs=state.rngs, apply_fn=apply_fn,
        loss_fn=loss_with_auxiliary_attack_fn, max_epsilon=max_epsilon,
        num_updates=num_updates)
    state = state.replace(rngs=rngs)
    return train_step_fn(state, images, labels, *args, **kwargs)

  return new_train_step_fn


def wrap_train_step_with_mixup(
    train_step_fn: TrainStepFn,
    *,
    concentration: Optional[float] = None,
    granularity: str = 'device',
    size: int = 2,
    partition_spec: Optional[PartitionSpec] = None,
) -> TrainStepFn:
  """Wraps a train step function with mixup.

  Args:
    train_step_fn: Training step function to wrap.
    concentration: Dirichlet concentration parameter, used to sample the
      mixing weights. Must be a positive float.
    granularity: Granularity of the mixing weights. 'batch' samples a single set
      of weights for all the examples in the batch, 'device' samples a set of
      weights for all the examples in each device, and 'example' samples weights
      for each example independently.
    size: Number of original examples used in each new mixed example.
    partition_spec: Optional PartitionSpec used to annotate the sampled values
      from the Dirichlet distribution. Only used if granularity is 'example'.

  Returns:
    A new train_step_fn that does mixup before calling the original one.
  """
  if concentration is None or concentration == 0.:
    return train_step_fn

  if granularity == 'batch':
    partition_spec = jax.sharding.PartitionSpec()

  if granularity == 'example':
    logging.warning(
        "You are using granularity = 'example'. This is extremely inneficient "
        'since the current implementation of jax.random.gamma does not work '
        "with pjit's sharding. Use granularity = 'device' meanwhile.")

  def new_train_step_fn(
      state: TrainState,
      images: Array,
      labels: Array,
      *args,
      **kwargs,
  ) -> Tuple[TrainState, Mapping[str, Any]]:
    if 'mixup' not in state.rngs:
      raise ValueError(f'Mixup requires a PRNGKey with the name "mixup", but '
                       f'only these were given: {sorted(state.rngs.keys())}')
    frozen_rngs = isinstance(state.rngs, flax.core.FrozenDict)
    rngs = flax.core.unfreeze(state.rngs) if frozen_rngs else state.rngs
    # Apply mixup on images and labels.
    mixup_rng = rngs.pop('mixup')
    mixup_rng, next_mixup_rng = jax.random.split(mixup_rng)
    # Reshape images and labels to perform mixup on each device independently,
    # to avoid communication among devices due to mixup's jnp.roll().
    images = images.reshape((jax.device_count(), -1) + images.shape[1:])
    labels = labels.reshape((jax.device_count(), -1) + labels.shape[1:])
    if granularity == 'batch':
      mixup_shape = (1, 1, size)
    elif granularity == 'device':
      mixup_shape = (images.shape[0], 1, size)
    elif granularity == 'example':
      mixup_shape = (*images.shape[:2], size)
    else:
      raise ValueError(f'Unknown granularity: {granularity!r}')
    (images, labels) = mixup(
        rng=mixup_rng, tree=(images, labels), concentration=concentration,
        shape=mixup_shape, roll_axis=1, partition_spec=partition_spec)
    images = images.reshape((-1,) + images.shape[2:])
    labels = labels.reshape((-1,) + labels.shape[2:])
    # Call the given `train_step_fn` passing the state without mixup rngs, and
    # the modified images and labels.
    state = state.replace(rngs=flax.core.freeze(rngs) if frozen_rngs else rngs)
    state, metrics = train_step_fn(state, images, labels, *args, **kwargs)
    # Add the mixup PRNGKey back to the state.
    rngs = flax.core.unfreeze(state.rngs) if frozen_rngs else state.rngs
    rngs['mixup'] = next_mixup_rng
    state = state.replace(rngs=flax.core.freeze(rngs) if frozen_rngs else rngs)
    return state, metrics

  return new_train_step_fn
