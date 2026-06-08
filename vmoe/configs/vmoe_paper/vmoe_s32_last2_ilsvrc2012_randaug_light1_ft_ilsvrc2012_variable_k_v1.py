# Variable-K routing V1 experiment config.

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
  config.description = config.description + ', Variable-K V1'
  _clear_router_adapter(config)
  config.model.encoder.moe.router.variable_k = ml_collections.ConfigDict({
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
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))
