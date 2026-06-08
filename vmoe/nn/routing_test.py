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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from vmoe.nn import routing


class NoisyTopExpertsPerItemRouterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # We mock get_top_items_per_expert_dispatcher to avoid having to specify the
    # parameters of the dispatcher during testing. The output of the
    # NoisyTopItemsPerExpertRouter is supposed to be a dispatcher, but we will
    # simply return the `gates_softmax`, which is fine for testing purposes.
    self.mock_get_top_items_per_expert_dispatcher = self.enter_context(
        mock.patch.object(
            routing.vmoe.moe,
            'get_top_experts_per_item_dispatcher',
            side_effect=lambda x, **_: x))

  def test_gshard_auxiliary_loss(self):
    gates = jnp.asarray([[.5, .4, .1], [.3, .3, .4], [.1, .2, .7],
                         [.8, .2, .0]])
    output = routing.NoisyTopExpertsPerItemRouter._gshard_auxiliary_loss(gates)
    # mean_gates_per_expert = [1.7, 1.1, 1.2] / 4.
    # mean_top1_per_expert = ([1, 0, 0]+[0, 0, 1]+[0, 0, 1]+[1, 0, 0]) / 4 =
    # [2, 0, 2] / 4.
    expected_output = 1.0875
    self.assertAlmostEqual(expected_output, output, places=6)

  def test_importance_auxiliary_loss(self):
    gates = jnp.asarray([[.5, .4, .1], [.3, .3, .4], [.1, .2, .7],
                         [.8, .2, .0]])
    output = routing.NoisyTopExpertsPerItemRouter._importance_auxiliary_loss(
        gates)
    # sum_gates_per_expert = [1.7, 1.1, 1.2]
    # mean(sum_gates_per_expert) = 1.3333334
    # std(sum_gates_per_expert) = 0.2624669
    # coefficient of variation = 0.19685018
    # coefficient of variation ** 2 = 0.03874999
    expected_output = 0.03874999
    self.assertAlmostEqual(expected_output, output, places=6)

  def test_load_auxiliary_loss(self):
    # batch_size = 3, num_experts = 3, k = 2.
    logits = jnp.asarray([[.9, .06, .04], [.7, .18, .12], [.85, .05, .1]],
                         dtype=jnp.float32)
    noise = jnp.asarray(
        [[-.037, .026, -.018], [-.074, .045, -.015], [-.067, -.059, .073]],
        dtype=jnp.float32)
    logits_noisy = logits + noise
    output = routing.NoisyTopExpertsPerItemRouter._load_auxiliary_loss(
        logits, logits_noisy, noise_std=.1, num_selected_experts=2)
    # In this case there's a clear winner (first expert) which is selected whp.
    # This increases the variance across experts and, thus, the auxiliary loss.
    # p_mean = [0.999 0.277 0.234]
    # std_p_mean = 0.3512197
    # mean_p_mean = 0.5039385
    # coefficient of variation = 0.6969495
    # coefficient of variation ** 2 = 0.48573864536489236
    expected_output = 0.48573864536489236
    self.assertAlmostEqual(expected_output, float(output), places=6)

  def test_variable_k_subset_action_helpers(self):
    self.assertEqual(routing.get_variable_k_num_subset_actions(5), 16)
    self.assertEqual(routing.variable_k_base_action(base_k=3, candidate_m=5), 3)
    actions = jnp.asarray([0, 1, 3, 15], dtype=jnp.int32)
    masks = routing.variable_k_action_to_selection_mask(actions, candidate_m=5)
    expected = jnp.asarray([
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])
    chex.assert_trees_all_equal(masks, expected)

  # We mock get_top_experts_per_item_dispatcher to avoid having to specify the
  # parameters of the dispatcher during testing. The output of the
  # NoisyTopExpertsPerItemRouter is supposed to be a dispatcher, but we will
  # simply return the `gates_softmax`, which is fine for testing purposes.

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=1.0,
        deterministic=True)
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    chex.assert_trees_all_close(y1, y2)
    chex.assert_trees_all_close(m1, m2)

  def test_variable_k_deterministic_zero_init_selects_base_k(self):
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=0.0,
        deterministic=True,
        variable_k={
            'enabled': True,
            'candidate_m': 3,
            'base_k': 2,
            'min_selected': 1,
            'hidden_dim': 8,
            'expert_embedding_dim': 4,
            'rank_embedding_dim': 2,
        })
    output, _ = layer.init_with_output(
        {'params': jax.random.PRNGKey(0), 'gating': jax.random.PRNGKey(1)}, x)
    y, metrics = output
    self.assertEqual(y.shape, (1, 5, 4))
    chex.assert_trees_all_close(
        jnp.sum(y > 0, axis=-1), jnp.full((1, 5), 2))
    self.assertIn('routing/variable_k/avg_selected_k', metrics)
    self.assertIn('routing/variable_k/importance_weighted_miss', metrics)
    self.assertIn('routing/variable_k/policy_entropy', metrics)
    self.assertAlmostEqual(
        float(metrics['routing/variable_k/avg_selected_k']), 2.0, places=5)
    self.assertTrue(jnp.isfinite(metrics['routing/variable_k/policy_log_prob']))

  def test_variable_k_forced_action_selects_subset_only(self):
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=0.0,
        deterministic=True,
        variable_k={
            'enabled': True,
            'candidate_m': 3,
            'base_k': 2,
            'min_selected': 1,
            'hidden_dim': 8,
            'expert_embedding_dim': 4,
            'rank_embedding_dim': 2,
        })
    variables = layer.init(
        {'params': jax.random.PRNGKey(0), 'gating': jax.random.PRNGKey(1)}, x)
    routing_context = {
        'variable_k/action_ids': jnp.zeros((1, 1, 5), dtype=jnp.int32),
    }
    y, metrics = layer.apply(
        variables, x, routing_context=routing_context,
        rngs={'gating': jax.random.PRNGKey(2)})
    chex.assert_trees_all_close(
        jnp.sum(y > 0, axis=-1), jnp.ones((1, 5)))
    self.assertAlmostEqual(
        float(metrics['routing/variable_k/avg_selected_k']), 1.0, places=5)

  def test_forward_not_deterministic(self):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopExpertsPerItemRouter(
        num_experts=4,
        num_selected_experts=2,
        noise_std=2.0,
        deterministic=False)
    # y's are dispatch weights, m's are metrics.
    y1, m1 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, m2 = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    different_fn = lambda x, y: jnp.abs(x - y).sum() > 0.01
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)
    # Importance loss and pre-dispatch router metrics are computed before
    # adding noise, so only noisy routing losses are expected to differ.
    chex.assert_trees_all_close(m1['importance_loss'], m2['importance_loss'])
    self.assertTrue(different_fn(m1['gshard_loss'], m2['gshard_loss']))
    self.assertTrue(different_fn(m1['load_loss'], m2['load_loss']))
    self.assertTrue(different_fn(m1['auxiliary_loss'], m2['auxiliary_loss']))


class NoisyTopItemsPerExpertRouterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # We mock get_top_items_per_expert_dispatcher to avoid having to specify the
    # parameters of the dispatcher during testing. The output of the
    # NoisyTopItemsPerExpertRouter is supposed to be a dispatcher, but we will
    # simply return the `gates_softmax`, which is fine for testing purposes.
    self.mock_get_top_items_per_expert_dispatcher = self.enter_context(
        mock.patch.object(
            routing.vmoe.moe, 'get_top_items_per_expert_dispatcher',
            side_effect=lambda x, **_: (x, {})))

  def test_forward_deterministic(self):
    """Tests that output is the same given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=1.0,
        deterministic=True)
    # y's are dispatch weights.
    y1, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    chex.assert_trees_all_close(y1, y2)

  def test_forward_not_deterministic(self):
    """Tests that output is different given two different gating PRNG seeds."""
    x = jnp.arange(5 * 4).reshape(1, 5, 4).astype(jnp.float32)
    variables = {'params': {'dense': {'kernel': jnp.eye(4)}}}
    layer = routing.NoisyTopItemsPerExpertRouter(
        num_experts=4,
        noise_std=1.0,
        deterministic=False)
    # y's are dispatch weights.
    y1, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(0)})
    y2, _ = layer.apply(variables, x, rngs={'gating': jax.random.PRNGKey(1)})
    different_fn = lambda x, y: jnp.abs(x - y).sum() > 0.01
    error_msg_fn = lambda x, y: f'{x} is too close to {y}'
    chex.assert_trees_all_equal_comparator(different_fn, error_msg_fn, y1, y2)


if __name__ == '__main__':
  absltest.main()
