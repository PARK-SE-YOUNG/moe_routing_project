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

"""Module with routing layers."""
import functools
from typing import Any, Mapping, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import vmoe.moe

Array = jnp.ndarray
BaseDispatcher = vmoe.moe.BaseDispatcher
DType = type(jnp.float32)
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]

class RouterAdapter(nn.Module):
  """Small adapter that predicts delta logits for router fine-tuning.

  If use_context_features=True, lightweight summary features derived from the
  current router input are broadcast and concatenated to each token before the
  adapter MLP. This is a scaffold for context-aware routing while keeping the
  original router logits unchanged.
  """
  num_experts: int
  hidden_dim: int = 64
  use_context_features: bool = False
  dtype: Optional[DType] = None

  def _make_context_features(
      self,
      inputs: Array,
      routing_context: Optional[Mapping[str, Array]] = None,
  ) -> Array:
    """Creates broadcast context features for each token.

    The first scaffold only uses input-derived statistics because they are JAX
    arrays available inside the router. Host-side GPU stats remain logging-only.
    """
    del routing_context

    token_l2 = jnp.linalg.norm(inputs, axis=-1, keepdims=True)
    token_abs_mean = jnp.mean(jnp.abs(inputs), axis=-1, keepdims=True)
    token_abs_max = jnp.max(jnp.abs(inputs), axis=-1, keepdims=True)

    group_l2_mean = jnp.mean(token_l2, axis=1, keepdims=True)
    group_abs_mean = jnp.mean(token_abs_mean, axis=1, keepdims=True)
    group_abs_max = jnp.max(token_abs_max, axis=1, keepdims=True)

    group_l2_mean = jnp.broadcast_to(group_l2_mean, token_l2.shape)
    group_abs_mean = jnp.broadcast_to(group_abs_mean, token_abs_mean.shape)
    group_abs_max = jnp.broadcast_to(group_abs_max, token_abs_max.shape)

    return jnp.concatenate(
        [
            token_l2,
            token_abs_mean,
            token_abs_max,
            group_l2_mean,
            group_abs_mean,
            group_abs_max,
        ],
        axis=-1,
    )

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      routing_context: Optional[Mapping[str, Array]] = None,
  ) -> Array:
    dtype = self.dtype or inputs.dtype
    adapter_inputs = inputs

    if self.use_context_features:
      context_features = self._make_context_features(
          inputs, routing_context=routing_context)
      context_features = context_features.astype(dtype)
      adapter_inputs = jnp.concatenate([inputs, context_features], axis=-1)

    x = nn.Dense(
        features=self.hidden_dim,
        dtype=dtype,
        name="fc1",
    )(adapter_inputs)
    x = nn.gelu(x)

    delta_logits = nn.Dense(
        features=self.num_experts,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.zeros,
        dtype=dtype,
        name="fc2",
    )(x)

    return delta_logits

class NoisyTopExpertsPerItemRouter(nn.Module):
  """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2106.05974.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (item, expert). Noise is added to these logits. The logits are normalized
  using a softmax over the expert dimension. This score will be used to
  determine which items are dispatched to which experts and how the outputs of
  the experts are combined.

  Because the routing algorithm is non-differentiable, the only way to train the
  parameters of the dense (a.k.a. gating layer) is through the weights used
  to combine the output of the experts, and through two auxiliary losses that
  depend on the output of the gating.
  """
  num_experts: int
  num_selected_experts: int = 1
  noise_std: float = 1.0
  gshard_loss_weight: float = 0.0
  importance_loss_weight: float = 1.0
  load_loss_weight: float = 1.0
  dispatcher: Optional[KwArgs] = None
  adapter: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      routing_context: Optional[Mapping[str, Array]] = None,
  ) -> Tuple[BaseDispatcher, Metrics]:
    gates_softmax, metrics = self._compute_gates_softmax_and_metrics(
        inputs, self.num_experts, routing_context)
    dispatcher = self._create_dispatcher(gates_softmax)
    metrics = {
        **metrics,
        **self._compute_dispatcher_metrics(dispatcher),
    }
    return dispatcher, metrics

  @nn.nowrap
  def _compute_dispatcher_metrics(self, dispatcher: BaseDispatcher) -> Metrics:
    """Computes capacity overflow / dropped assignment metrics from dispatcher.

    The TopExpertsPerItem dispatcher drops assignments whose expert buffer index
    exceeds the per-expert capacity. For the einsum dispatcher this appears as
    zero dispatch weights after one_hot(buffer_idx, capacity). For the indices
    dispatcher this appears as indices whose buffer position is >= capacity.
    """
    if isinstance(dispatcher, vmoe.moe.Bfloat16Dispatcher):
      return self._compute_dispatcher_metrics(dispatcher.dispatcher)

    if isinstance(dispatcher, vmoe.moe.EinsumDispatcher):
      dispatch_weights = (
          dispatcher.combine_weights > 0
          if dispatcher.dispatch_weights is None else dispatcher.dispatch_weights)
      kept_assignments = jnp.sum(dispatch_weights.astype(jnp.float32))
      num_groups = dispatch_weights.shape[0]
      group_size = dispatch_weights.shape[1]
      requested_assignments = jnp.asarray(
          num_groups * group_size * self.num_selected_experts,
          dtype=jnp.float32)
      dropped_ratio = 1.0 - kept_assignments / jnp.maximum(
          requested_assignments, 1.0)
      dropped_ratio = jnp.clip(dropped_ratio, 0.0, 1.0)
      return {
          "routing/overflow_ratio": dropped_ratio,
          "routing/dropped_token_ratio": dropped_ratio,
          "routing/kept_assignment_count": kept_assignments,
          "routing/requested_assignment_count": requested_assignments,
      }

    if isinstance(dispatcher, vmoe.moe.ExpertIndicesDispatcher):
      valid = jnp.logical_and(
          dispatcher.indices[..., 0] < dispatcher.num_experts,
          dispatcher.indices[..., 1] < dispatcher.capacity)
      valid = jnp.logical_and(valid, dispatcher.indices[..., 1] >= 0)
      kept_assignments = jnp.sum(valid.astype(jnp.float32))
      requested_assignments = jnp.asarray(valid.size, dtype=jnp.float32)
      dropped_ratio = 1.0 - kept_assignments / jnp.maximum(
          requested_assignments, 1.0)
      dropped_ratio = jnp.clip(dropped_ratio, 0.0, 1.0)
      return {
          "routing/overflow_ratio": dropped_ratio,
          "routing/dropped_token_ratio": dropped_ratio,
          "routing/kept_assignment_count": kept_assignments,
          "routing/requested_assignment_count": requested_assignments,
      }

    return {
        "routing/overflow_ratio": jnp.asarray(0.0, dtype=jnp.float32),
        "routing/dropped_token_ratio": jnp.asarray(0.0, dtype=jnp.float32),
    }

  @nn.nowrap
  def _compute_gates_softmax_and_metrics(
      self,
      inputs: Array,
      num_experts: int,
      routing_context: Optional[Mapping[str, Array]] = None,
  ) -> Tuple[Array, Metrics]:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    if not num_experts >= self.num_selected_experts >= 1:
      raise ValueError(f"num_experts >= num_selected_experts >= 1, but got "
                       f"num_experts = {num_experts} and "
                       f"num_selected_experts = {self.num_selected_experts}.")
    dtype = self.dtype or inputs.dtype

    # Compute the original router logits.
    gates_logits_original = nn.Dense(features=num_experts, use_bias=False,
                                     dtype=dtype, name="dense")(inputs)
    gates_logits = gates_logits_original

    # Optional RouterAdapter: logits_new = logits_original + Delta_theta(x, h, l).
    if self.adapter:
      adapter_kwargs = dict(**self.adapter)
      adapter_hidden_dim = adapter_kwargs.pop("hidden_dim", 64)
      adapter_use_context_features = adapter_kwargs.pop(
          "use_context_features", False)
      delta_logits = RouterAdapter(
          num_experts=num_experts,
          hidden_dim=adapter_hidden_dim,
          use_context_features=adapter_use_context_features,
          dtype=dtype,
          name="RouterAdapter",
          **adapter_kwargs,
      )(inputs, routing_context=routing_context)
      gates_logits = gates_logits_original + delta_logits

    # Router probabilities before optional noisy dispatch.
    gates_softmax = jax.nn.softmax(gates_logits)
    gates_softmax_original = jax.nn.softmax(gates_logits_original)

    selected_expert = jnp.argmax(gates_softmax, axis=-1)
    selected_log_prob = jnp.sum(
        jax.nn.one_hot(selected_expert, num_experts)
        * jnp.log(gates_softmax + 1e-8),
        axis=-1,
    ).mean()

    router_kl_to_original = jnp.sum(
        gates_softmax * (
            jnp.log(gates_softmax + 1e-8)
            - jnp.log(gates_softmax_original + 1e-8)
        ),
        axis=-1,
    ).mean()

    router_entropy = -jnp.sum(
        gates_softmax * jnp.log(gates_softmax + 1e-8),
        axis=-1,
    ).mean()

    router_confidence = jnp.max(
        gates_softmax,
        axis=-1,
    ).mean()

    top1_expert = jnp.argmax(gates_softmax, axis=-1)
    expert_usage = jnp.sum(
        jax.nn.one_hot(top1_expert, num_experts),
        axis=(0, 1),
    )

    expert_usage_min = expert_usage.min()
    expert_usage_max = expert_usage.max()
    expert_usage_std = expert_usage.std()

    token_l2 = jnp.linalg.norm(inputs, axis=-1)
    token_metrics = {
        "router_input/token_l2_mean": jnp.mean(token_l2),
        "router_input/token_l2_std": jnp.std(token_l2),
        "router_input/token_abs_mean": jnp.mean(jnp.abs(inputs)),
        "router_input/token_abs_max": jnp.max(jnp.abs(inputs)),
        "router_input/token_num_groups": jnp.asarray(inputs.shape[0], dtype=jnp.float32),
        "router_input/token_num_tokens": jnp.asarray(inputs.shape[1], dtype=jnp.float32),
        "router_input/token_hidden_dim": jnp.asarray(inputs.shape[2], dtype=jnp.float32),
    }

    router_context_metrics = {
        "router_context/expert_usage_min": expert_usage_min,
        "router_context/expert_usage_max": expert_usage_max,
        "router_context/expert_usage_std": expert_usage_std,
        "router_context/router_entropy": router_entropy,
        "router_context/router_confidence": router_confidence,
        "router_context/selected_log_prob": selected_log_prob,
        "router_context/router_kl_to_original": router_kl_to_original,
    }

    # routing/* namespace for WandB dashboards and future finite-horizon RL
    # routing analysis. Expert token counts are top-1 counts from the router
    # probabilities before optional noisy dispatch.
    routing_metrics = {
        "routing/expert_token_count_min": expert_usage_min,
        "routing/expert_token_count_max": expert_usage_max,
        "routing/expert_token_count_std": expert_usage_std,
        "routing/router_entropy": router_entropy,
        "routing/router_confidence": router_confidence,
        "routing/selected_log_prob": selected_log_prob,
        "routing/router_kl_to_original": router_kl_to_original,
        "routing/overflow_ratio": jnp.asarray(0.0, dtype=jnp.float32),
        "routing/dropped_token_ratio": jnp.asarray(0.0, dtype=jnp.float32),
    }
    routing_metrics.update({
        f"routing/expert_{i:02d}_token_count": expert_usage[i]
        for i in range(num_experts)
    })

    importance_loss = jax.vmap(self._importance_auxiliary_loss)(gates_softmax)
    load_loss = jnp.zeros_like(importance_loss)

    if self.deterministic or self.noise_std == 0.0:
      gshard_loss = jax.vmap(self._gshard_auxiliary_loss)(gates_softmax)
      auxiliary_loss = _weighted_sum(
          (self.gshard_loss_weight, gshard_loss),
          (self.importance_loss_weight, importance_loss),
          (self.load_loss_weight, load_loss))

      metrics = {
          "auxiliary_loss": auxiliary_loss,
          "gshard_loss": gshard_loss,
          "importance_loss": importance_loss,
          "load_loss": load_loss,
          "router_entropy": router_entropy,
          "router_confidence": router_confidence,
          "expert_usage_min": expert_usage_min,
          "expert_usage_max": expert_usage_max,
          "expert_usage_std": expert_usage_std,
          "selected_log_prob": selected_log_prob,
          "router_kl_to_original": router_kl_to_original,
          "routing/auxiliary_loss": auxiliary_loss,
          "routing/gshard_loss": gshard_loss,
          "routing/importance_loss": importance_loss,
          "routing/load_loss": load_loss,
          **token_metrics,
          **router_context_metrics,
          **routing_metrics,
      }
      return gates_softmax, metrics

    noise_std = (1.0 / num_experts) * self.noise_std
    logits_noise = noise_std * jax.random.normal(
        key=self.make_rng("gating"), shape=gates_logits.shape)
    gates_logits_noisy = gates_logits + logits_noise
    gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)

    load_loss = jax.vmap(  # pytype: disable=wrong-arg-types
        functools.partial(
            self._load_auxiliary_loss,
            num_selected_experts=self.num_selected_experts,
            noise_std=noise_std))(gates_logits, gates_logits_noisy)
    gshard_loss = jax.vmap(self._gshard_auxiliary_loss)(gates_softmax_noisy)
    auxiliary_loss = _weighted_sum(
        (self.gshard_loss_weight, gshard_loss),
        (self.importance_loss_weight, importance_loss),
        (self.load_loss_weight, load_loss))

    metrics = {
        "auxiliary_loss": auxiliary_loss,
        "gshard_loss": gshard_loss,
        "importance_loss": importance_loss,
        "load_loss": load_loss,
        "router_entropy": router_entropy,
        "router_confidence": router_confidence,
        "expert_usage_min": expert_usage_min,
        "expert_usage_max": expert_usage_max,
        "expert_usage_std": expert_usage_std,
        "selected_log_prob": selected_log_prob,
        "router_kl_to_original": router_kl_to_original,
        "routing/auxiliary_loss": auxiliary_loss,
        "routing/gshard_loss": gshard_loss,
        "routing/importance_loss": importance_loss,
        "routing/load_loss": load_loss,
        **token_metrics,
        **router_context_metrics,
        **routing_metrics,
    }
    return gates_softmax_noisy, metrics

  @nn.nowrap
  def _create_dispatcher(self, gates_dispatch):
    # Creates a dispatcher implementing the TopExpertsPerItem routing algorithm,
    # that uses at most `num_selected_experts` per item. Notice that each
    # group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_experts_per_item_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_experts_per_item_dispatcher,
            num_selected_experts=self.num_selected_experts,
            **dispatcher_kwargs))
    dispatcher = get_top_experts_per_item_dispatcher_vmapped(gates_dispatch)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher

  @classmethod
  def _gshard_auxiliary_loss(cls, gates: Array) -> Array:
    # See `l_{aux}` in Algorithm 1 in https://arxiv.org/pdf/2006.16668.pdf.
    _, num_experts = gates.shape
    # Line (3) in Algorithm 1.
    mean_gates_per_expert = gates.mean(axis=0)
    # Lines (11, 13) in Algorithm 1.
    mean_top1_per_expert = jax.nn.one_hot(
        jnp.argmax(gates, axis=1), num_experts, dtype=jnp.int32).mean(axis=0)
    # Note: Only gradients through mean_gates_per_expert affect the gating,
    # since hard counts from top_k+one_hot are not differentiable.
    auxiliary_loss = jnp.mean(mean_top1_per_expert * mean_gates_per_expert)
    # Note: Not mentioned in the paper, but it's done in their source code.
    # https://github.com/tensorflow/lingvo/blob/84b85514d7ad3652bc9720cb45acfab08604519b/lingvo/core/gshard_layers.py#L2223
    auxiliary_loss *= num_experts**2
    return auxiliary_loss

  @classmethod
  def _importance_auxiliary_loss(cls, gates: Array) -> Array:
    axis = tuple(range(gates.ndim - 1))  # All except last.
    importance_per_expert = jnp.sum(gates, axis=axis)
    std_importance_per_expert = jnp.std(importance_per_expert)
    mean_importance_per_expert = jnp.mean(importance_per_expert)
    # Compute coefficient of variation (i.e. std/mean) squared.
    return (std_importance_per_expert / mean_importance_per_expert)**2

  @classmethod
  def _load_auxiliary_loss(cls, logits: Array, logits_noisy: Array,
                           noise_std: Array,
                           num_selected_experts: int) -> Array:
    # For each example, compute the weight required for an expert to be selected
    # among the top-k.
    # NOTE: DO NOT TRY TO SIMPLIFY THIS. This convoluted way of obtaining the
    # threshold_per_item avoids adding all-gather ops during backpropagation.
    num_experts = logits_noisy.shape[-1]
    threshold_per_item_index = jax.lax.top_k(
        logits_noisy, num_selected_experts)[-1][..., -1]
    threshold_per_item = jnp.sum(
        jax.nn.one_hot(threshold_per_item_index, num_experts) * logits_noisy,
        axis=-1)
    # For each example and expert, find how far they were from the threshold and
    # normalize this value by the noise_std to use the standard Gaussian CDF.
    noise_required_to_win = threshold_per_item[..., None] - logits
    noise_required_to_win /= noise_std
    # p is the probability of being above the threshold for each (item, expert)
    # if the random noise (with its std) was re-sampled again.
    p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
    # We compute the average such probability for each expert over examples.
    p_mean = jnp.mean(p, axis=0)
    # Compute p_mean's coefficient of variation squared.
    return (jnp.std(p_mean) / jnp.mean(p_mean))**2


class NoisyTopItemsPerExpertRouter(nn.Module):
  """Noisy TopItemsPerExpert router.

  Instead of picking the Top-K experts with highest score for each item, and
  then ignore choices that exceed the capacity (C) of any given expert, here we
  pick the Top-C items with highest score for each expert.

  This makes the load across experts automatically balanced, however the number
  of experts assigned to each item is not bounded and can vary. Some items may
  not be routed to any expert. In practice, though, this works very well.

  This was coined "Experts Choice Routing" in https://arxiv.org/abs/2202.09368.
  """
  num_experts: int
  noise_std: float = 1.0
  dispatcher: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[BaseDispatcher, Metrics]:
    gates_softmax = self._compute_gates_softmax(inputs, self.num_experts)
    dispatcher, metrics = self._create_dispatcher_and_metrics(gates_softmax)
    metrics["auxiliary_loss"] = 0.
    return dispatcher, metrics  # pytype: disable=bad-return-type

  @nn.nowrap
  def _compute_gates_softmax(self, inputs: Array, num_experts: int) -> Array:
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    if self.deterministic or self.noise_std == 0.0:
      gates_softmax = jax.nn.softmax(gates_logits)
      return gates_softmax
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      logits_noise = noise_std * jax.random.normal(
          key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      return gates_softmax_noisy

  @nn.nowrap
  def _create_dispatcher_and_metrics(self, gates_dispatch):
    # Creates a dispatcher implementing the TopItemsPerExpert routing algorithm.
    # Notice that each group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_items_per_expert_dispatcher_vmapped = jax.vmap(
        functools.partial(
            vmoe.moe.get_top_items_per_expert_dispatcher, **dispatcher_kwargs))
    dispatcher, metrics = get_top_items_per_expert_dispatcher_vmapped(
        gates_dispatch)
    if use_bfloat16:
      dispatcher = vmoe.moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher, metrics


def _weighted_sum(*args):
  """Returns a weighted sum of [(weight, element), ...] for weights > 0."""
  # Note: some losses might be ill-defined in some scenarios (e.g. they may
  # have inf/NaN gradients), in those cases we don't apply them on the total
  # auxiliary loss, by setting their weights to zero.
  return sum(x * w for w, x in args if w > 0)
