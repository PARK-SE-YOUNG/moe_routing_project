# Variable-K routing V1 short smoke run on local RunPod data.

from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_variable_k_v1
    as base_config)


TFDS_DATA_DIR = '/workspace/imagenet/tfds'
TFDS_MANUAL_DIR = '/workspace/imagenet/raw'
BATCH_SIZE = 64


def _set_data_config(config, split):
  config.split = split
  config.batch_size = BATCH_SIZE
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  config.prefetch = 1
  config.prefetch_device = 1


def get_config():
  config = base_config.get_config()

  # Keep this config runnable on the current 2-GPU RunPod instance.
  config.train_steps = 20
  config.num_expert_partitions = 2

  _set_data_config(config.dataset.train, 'validation[:128]')
  _set_data_config(config.dataset.val, 'validation[:64]')
  _set_data_config(config.dataset.test, 'validation[:64]')
  del config.dataset.imagenet_v2
  del config.dataset.test_real

  config.optimizer.learning_rate.warmup_steps = 5
  config.report_progress.every_steps = 1
  config.evaluate.every_steps = 20
  config.save_checkpoint.every_steps = 10_000_000
  config.disable_checkpoint_save = True
  config.profile.first_profile = 10_000_000

  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', [0])
