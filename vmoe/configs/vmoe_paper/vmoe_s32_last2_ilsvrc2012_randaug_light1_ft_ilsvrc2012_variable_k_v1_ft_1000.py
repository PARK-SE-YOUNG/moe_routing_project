# Variable-K V1 adapter training from the official fine-tuned E=8 checkpoint.

import ml_collections
from vmoe.configs.vmoe_paper import common


TFDS_MANUAL_DIR = '/workspace/imagenet/raw'
TFDS_DATA_DIR = '/workspace/imagenet/tfds'
BATCH_SIZE = 64
NUM_CLASSES = 1_000
IMAGE_SIZE = 384


def get_config():
  config = common.get_base_config()

  config.dataset = ml_collections.ConfigDict()
  pp_common = (
      f'value_range(-1,1)|onehot({NUM_CLASSES}, inkey="label", '
      'outkey="labels")|keep("image", "labels")')
  config.dataset.train = get_data_train_config(
      name='imagenet2012',
      split='train[:64000]',
      process=(
          f'decode_jpeg_and_inception_crop({IMAGE_SIZE})|flip_lr|'
          f'{pp_common}'))
  config.dataset.val = get_data_eval_config(
      name='imagenet2012',
      split='train[99%:]',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}')
  config.dataset.test = get_data_eval_config(
      name='imagenet2012',
      split='validation',
      process=f'decode|resize({IMAGE_SIZE})|{pp_common}')

  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  config.train_steps = 1000
  config.evaluate.every_steps = 250
  config.report_progress.every_steps = 25
  config.save_checkpoint.every_steps = 10_000_000
  config.save_checkpoint.keep_last = 0
  config.disable_checkpoint_save = True
  config.metric_logging = ml_collections.ConfigDict({
      'compact_train_metrics': True,
      # Training-time throughput/FLOPs are intentionally omitted here. The
      # router latency question should be answered by inference/eval runs.
      'log_train_cost': False,
      'train_allowlist': (
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
      ),
  })
  config.report_progress.write_progress_metrics = False

  config.description = 'ViT-S/32, E=8, K=2, Last 2, 300 Epochs'
  config.model = get_vmoe_config(config.description)
  config.description = config.description + ', Variable-K V1 FT 1000'

  config.initialization = ml_collections.ConfigDict({
      'name': 'initialize_from_vmoe',
      'prefix':
          'gs://vmoe_checkpoints/vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012',
      'rules': [
          ('VariableKSubsetPolicyAdapter/.*', ''),
          ('pre_logits/.*', ''),
          ('^(.*/pos_embedding)$', r'params/\1', 'vit_zoom'),
          ('^(.*)$', r'params/\1'),
      ],
      'raise_if_target_unmatched': False,
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
          'warmup_steps': 100,
      },
      'gradient_clip': {'global_norm': 10.0},
  })
  config.optimizer.trainable_pattern = 'VariableKSubsetPolicyAdapter'

  config.ppo = ml_collections.ConfigDict({
      'enabled': True,
      'miss_reward_weight': 1.0,
      'usage_entropy_weight': 0.01,
      'latency_weight': 0.01,
      'latency_baseline_secs': 0.052,
      'clip_coef': 0.2,
      'vf_coef': 0.5,
      'ent_coef': 0.01,
      'update_epochs': 2,
      'num_minibatches': 1,
      'advantage_normalize': True,
  })

  config.rl_loss = ml_collections.ConfigDict({
      'miss_weight': 1.0,
      'usage_entropy_weight': 0.01,
      'selection_budget_weight': 0.0,
  })

  config.num_expert_partitions = 2
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating')

  return config


def get_data_train_config(name, split, process):
  config = common.get_data_config(
      name=name,
      split=split,
      process=process,
      batch_size=BATCH_SIZE,
      shuffle_buffer=4096,
      cache=None)
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 4
  config.prefetch_device = 1
  return config


def get_data_eval_config(name, split, process):
  config = common.get_data_config(
      name=name,
      split=split,
      process=process,
      batch_size=BATCH_SIZE,
      shuffle_buffer=None,
      cache=None)
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 1
  config.prefetch_device = 1
  return config


def get_vmoe_config(description: str) -> ml_collections.ConfigDict:
  config = common.get_vmoe_config(description, IMAGE_SIZE, NUM_CLASSES)
  config.representation_size = None
  config.encoder.moe.router.dispatcher.capacity_factor = 1.5
  config.encoder.moe.router.variable_k = ml_collections.ConfigDict({
      'enabled': True,
      'mode': 'learned',
      'candidate_m': 5,
      'base_k': 2,
      'base_logit_margin': 6.0,
      'min_selected': 1,
      'hidden_dim': 64,
      'expert_embedding_dim': 16,
      'rank_embedding_dim': 8,
  })
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', [0])
