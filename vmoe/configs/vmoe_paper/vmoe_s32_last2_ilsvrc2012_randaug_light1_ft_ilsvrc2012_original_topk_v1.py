# Original V-MoE top-K baseline wrapper for V1 comparisons.

from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012 as base_config)


def get_config():
  config = base_config.get_config()
  config.description = config.description + ', Original Top-K V1 Baseline'
  config.disable_checkpoint_save = True
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))
