# Zero-padded top-K inside top-M candidate framework.

import ml_collections
from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_router_adapter_smoke
    as base_config)


def _clear_router_adapter(config):
  router = config.model.encoder.moe.router
  if 'adapter' in router:
    del router.adapter


def get_config():
  config = base_config.get_config()
  config.description = config.description + ', Zero-padded Top-K-to-Top-M V1'
  _clear_router_adapter(config)
  config.model.encoder.moe.router.variable_k = ml_collections.ConfigDict({
      'enabled': True,
      'mode': 'top_k',
      'candidate_m': 5,
      'base_k': 2,
      'min_selected': 1,
  })
  config.optimizer.trainable_pattern = 'VariableKSubsetPolicyAdapter'
  config.optimizer.learning_rate.peak_value = 0.0
  config.optimizer.learning_rate.end_value = 0.0
  config.rl_loss = ml_collections.ConfigDict({
      'miss_weight': 1.0,
      'usage_entropy_weight': 0.01,
      'selection_budget_weight': 0.0,
  })
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))
