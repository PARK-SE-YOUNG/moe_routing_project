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

from absl.testing import absltest
import jax.numpy as jnp
from vmoe.train import cleanrl_ppo


class CleanRLPPOTest(absltest.TestCase):

  def test_categorical_log_prob_and_entropy(self):
    logits = jnp.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
    actions = jnp.asarray([0, 1], dtype=jnp.int32)
    log_prob = cleanrl_ppo.categorical_log_prob(logits, actions)
    entropy = cleanrl_ppo.categorical_entropy(logits)
    self.assertEqual(log_prob.shape, (2,))
    self.assertEqual(entropy.shape, (2,))
    self.assertTrue(jnp.all(jnp.isfinite(log_prob)))
    self.assertTrue(jnp.all(jnp.isfinite(entropy)))

  def test_clipped_ppo_loss_is_finite(self):
    old_log_prob = jnp.log(jnp.asarray([0.4, 0.5, 0.6], dtype=jnp.float32))
    new_log_prob = jnp.log(jnp.asarray([0.5, 0.4, 0.9], dtype=jnp.float32))
    old_values = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
    new_values = jnp.asarray([0.2, 0.1, 0.6], dtype=jnp.float32)
    returns = jnp.asarray([0.3, 0.0, 0.8], dtype=jnp.float32)
    advantages = returns - old_values
    entropy = jnp.asarray([0.7, 0.6, 0.5], dtype=jnp.float32)
    loss, metrics = cleanrl_ppo.clipped_ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        new_values=new_values,
        old_values=old_values,
        returns=returns,
        entropy=entropy,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01)
    self.assertTrue(jnp.isfinite(loss))
    for key in (
        'ppo/policy_loss', 'ppo/value_loss', 'ppo/entropy_loss',
        'ppo/approx_kl', 'ppo/clipfrac'):
      self.assertIn(key, metrics)
      self.assertTrue(jnp.isfinite(metrics[key]))


if __name__ == '__main__':
  absltest.main()
