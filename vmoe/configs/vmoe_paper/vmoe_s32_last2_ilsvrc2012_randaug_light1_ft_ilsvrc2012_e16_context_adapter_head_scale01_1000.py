# Context-aware RouterAdapter + head scale=0.1 E=16 1000-step matched run.
# Accuracy baseline config generated from E=16 smoke config.
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

# pylint: disable=line-too-long
r"""Fine-tune gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1 on ILSVRC2012 with three seeds.

Important disclaimer for comparisons: Although we named this backbone architecture
ViT-S, it is different from the ViT-S described in https://arxiv.org/abs/2106.10270
and https://arxiv.org/abs/2106.04560. Most notably, we have only 8 layers, while
these works use 12 layers in the ViT-S backbone. Unfortunately, all these were
concurrent works and we used the same name for slightly different things.

Fine-tuning takes about 48m on a TPUv3-8.
Accuracy (mean over 3 runs with different fine-tuning seeds):
  - ILSVRC2012, validation: 78.8%
  - ILSVRC2012, test: 75.9%
  - ILSVRC2012 ReaL, test: 77.1%
  - ImageNet V2, test: 62.0%

"""
# pylint: enable=line-too-long
import ml_collections
from vmoe.configs.vmoe_paper import common

# Paths to manually downloaded datasets and to the tensorflow_datasets data dir.
TFDS_MANUAL_DIR = '/workspace/imagenet/raw'
TFDS_DATA_DIR = '/workspace/imagenet/tfds'
# The following configuration was made to fit on TPUv3-32. The number of images
# per device has to be at least 32.
BATCH_SIZE = 64    # Number of images processed in each step.
NUM_CLASSES = 1_000  # Number of ILSVRC2012 classes.
IMAGE_SIZE = 384     # Image size as input to the model.


def get_config():
  """Fine-tune gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1 on ILSVRC2012."""
  config = common.get_base_config()

  config.dataset = ml_collections.ConfigDict()
  pp_common = f'value_range(-1,1)|onehot({NUM_CLASSES}, inkey="label", outkey="labels")|keep("image", "labels")'
  # Smoke fine-tuning dataset.
  # Use a small train subset to verify that the new classification head learns.
  config.dataset.train = get_data_train_config(
      name='imagenet2012', split='train',
      process=(
          f'decode_jpeg_and_inception_crop({IMAGE_SIZE})|flip_lr|{pp_common}'))
  # Small validation subset for fast sanity accuracy check.
  config.dataset.test = get_data_eval_config(
      name='imagenet2012', split='validation',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}')
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  # Fine-tuning steps.
  config.train_steps = 1000
  config.evaluate.every_steps = 25
  config.save_checkpoint.every_steps = 10_000_000
  config.save_checkpoint.keep_last = 0
  # Description of the upstream model to fine-tune.
  config.description = 'ViT-S/32, E=8, K=2, Last 2, 300 Epochs'
  config.model = get_vmoe_config(config.description)
  # E=16 smoke test: 4 GPUs, 4 experts per GPU.
  config.model.encoder.moe.num_experts = 16
  config.num_expert_partitions = 4
  # Model initialization from the released checkpoints.
  config.initialization = ml_collections.ConfigDict({
      'name': 'initialize_from_vmoe',
      'prefix':
          'gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1',
      'rules': [
          ('head', ''),              # Do not restore the head params.
          ('Encoder/encoderblock_5/Moe/.*', ''),  # E=8 -> E=16 shape mismatch.
          ('Encoder/encoderblock_7/Moe/.*', ''),  # E=8 -> E=16 shape mismatch.
          ('pre_logits/.*', ''),     # Do not restore missing pre_logits params.
          # We pre-trained on 224px and are finetuning on 384px.
          # Resize positional embeddings.
          ('^(.*/pos_embedding)$', r'params/\1', 'vit_zoom'),
          # Restore the rest of parameters without any transformation.
          ('^(.*)$', r'params/\1'),
      ],
      # We are not initializing several arrays from the new train state, do not
      # raise an exception.
      'raise_if_target_unmatched': False,
      # Partition MoE parameters when reading from the checkpoint.
      'axis_resources_regexes': [('Moe/Mlp/.*', ('expert',))],
  })
  config.optimizer = ml_collections.ConfigDict({
      'name': 'sgd',
      'momentum': 0.9,
      'accumulator_dtype': 'float32',
      'learning_rate': {
          'schedule': 'warmup_cosine_decay',
          'peak_value': 0.001,
          'end_value': 0.0,
          'warmup_steps': 10,
      },
      'gradient_clip': {'global_norm': 10.0},
      })
  
  config.optimizer.trainable_pattern = 'RouterAdapter|head'
  

  # These control how the model parameters are partitioned across the device
  # mesh for running the models efficiently.
  # By setting num_expert_partitions = num_experts, we set at most one expert on
  # each device.
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  # This value specifies that the first axis of all parameters in the MLPs of
  # MoE layers (which has size num_experts) is partitioned across the 'expert'
  # axis of the device mesh.
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating')

  # Smoke runs skip checkpoint saving to avoid Orbax save issues.
  config.disable_checkpoint_save = True
  return config


def get_data_train_config(name, split, process):
  """Returns dataset parameters."""
  config = common.get_data_config(
      name=name, split=split, process=process, batch_size=BATCH_SIZE,
      shuffle_buffer=50_000, cache=None)
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 16
  config.prefetch_device = 1
  return config


def get_data_eval_config(name, split, process):
  config = common.get_data_config(
      name=name, split=split, process=process, batch_size=BATCH_SIZE,
      shuffle_buffer=None, cache='batched')
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 1
  config.prefetch_device = 1
  return config


def get_vmoe_config(description: str) -> ml_collections.ConfigDict:
  config = common.get_vmoe_config(description, IMAGE_SIZE, NUM_CLASSES)
  config.representation_size = None
  config.encoder.moe.router.dispatcher.capacity_factor = 1.5
  config.encoder.moe.router.adapter = ml_collections.ConfigDict({
      'hidden_dim': 64,
      'use_context_features': True,
      'scale': 0.1,
  })
  return config

def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))

