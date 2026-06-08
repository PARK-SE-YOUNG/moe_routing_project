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

"""Minimal CleanRL-style PPO loss utilities.

This module vendors the PPO objective structure used by CleanRL's PPO
implementations: clipped policy ratio loss, clipped value loss, policy entropy,
approximate KL, and clip fraction. The environment interaction is intentionally
not included because variable-K routing is a contextual-bandit problem rather
than a Gym rollout.
"""

from typing import Mapping, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def categorical_log_prob(logits: Array, actions: Array) -> Array:
  """Returns log pi(action | logits) for categorical actions."""
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  return jnp.take_along_axis(
      log_probs, actions.astype(jnp.int32)[..., None], axis=-1)[..., 0]


def categorical_entropy(logits: Array) -> Array:
  """Returns categorical entropy for each leading item."""
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  probs = jnp.exp(log_probs)
  return -jnp.sum(probs * log_probs, axis=-1)


def normalize_advantages(advantages: Array, eps: float = 1e-8) -> Array:
  """Normalizes advantages over all provided routing decisions."""
  return (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + eps)


def clipped_ppo_loss(
    *,
    new_log_prob: Array,
    old_log_prob: Array,
    advantages: Array,
    new_values: Array,
    old_values: Array,
    returns: Array,
    entropy: Array,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    normalize_advantage: bool = True,
) -> Tuple[Array, Mapping[str, Array]]:
  """Computes CleanRL-style clipped PPO loss and diagnostics."""
  old_log_prob = jax.lax.stop_gradient(old_log_prob)
  old_values = jax.lax.stop_gradient(old_values)
  returns = jax.lax.stop_gradient(returns)
  advantages = jax.lax.stop_gradient(advantages)
  if normalize_advantage:
    advantages = normalize_advantages(advantages)

  logratio = new_log_prob - old_log_prob
  ratio = jnp.exp(logratio)
  pg_loss1 = -advantages * ratio
  pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
  policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

  unclipped_value_loss = jnp.square(new_values - returns)
  value_clipped = old_values + jnp.clip(
      new_values - old_values, -clip_coef, clip_coef)
  clipped_value_loss = jnp.square(value_clipped - returns)
  value_loss = 0.5 * jnp.mean(jnp.maximum(
      unclipped_value_loss, clipped_value_loss))

  entropy_loss = jnp.mean(entropy)
  loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

  approx_kl = jnp.mean((ratio - 1.0) - logratio)
  old_approx_kl = -jnp.mean(logratio)
  clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))
  metrics = {
      'ppo/policy_loss': policy_loss,
      'ppo/value_loss': value_loss,
      'ppo/entropy_loss': entropy_loss,
      'ppo/approx_kl': approx_kl,
      'ppo/old_approx_kl': old_approx_kl,
      'ppo/clipfrac': clipfrac,
      'ppo/ratio_mean': jnp.mean(ratio),
      'ppo/advantage_mean': jnp.mean(advantages),
      'ppo/return_mean': jnp.mean(returns),
      'ppo/value_mean': jnp.mean(new_values),
  }
  return loss, metrics
