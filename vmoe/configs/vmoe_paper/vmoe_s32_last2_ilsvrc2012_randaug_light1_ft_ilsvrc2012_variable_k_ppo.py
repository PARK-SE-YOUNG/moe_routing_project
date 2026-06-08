# RL Variable-K categorical-subset PPO experiment config.

from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_variable_k_v1
    as base_config)


def get_config():
  config = base_config.get_config()
  config.description = config.description.replace(
      'Variable-K V1', 'Variable-K PPO')
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))
