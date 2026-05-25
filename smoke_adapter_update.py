import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict

from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012 as cfg
)
from vmoe.nn.vit_moe import VisionTransformerMoe
from vmoe.train import optimizer

from flax.core import unfreeze, freeze

c = cfg.get_config()

model = VisionTransformerMoe(**c.model)

x = jnp.ones((8, 384, 384, 3), jnp.float32)

labels = jax.nn.one_hot(
    jnp.zeros((8,), dtype=jnp.int32),
    1000,
)

rngs = {
    "params": jax.random.PRNGKey(0),
    "gating": jax.random.PRNGKey(1),
    "dropout": jax.random.PRNGKey(2),
}

variables = model.init(rngs, x)

from flax.core import unfreeze, freeze

variables = unfreeze(variables)

variables["params"]["Encoder"]["encoderblock_5"]["Moe"]["Router"]["RouterAdapter"]["fc2"]["kernel"] = (
    variables["params"]["Encoder"]["encoderblock_5"]["Moe"]["Router"]["RouterAdapter"]["fc2"]["kernel"]
    + 1e-3
)

variables["params"]["Encoder"]["encoderblock_7"]["Moe"]["Router"]["RouterAdapter"]["fc2"]["kernel"] = (
    variables["params"]["Encoder"]["encoderblock_7"]["Moe"]["Router"]["RouterAdapter"]["fc2"]["kernel"]
    + 1e-3
)

variables = freeze(variables)

params = variables["params"]

def loss_fn(params):

    flat = flatten_dict(params, sep="/")

    adapter_l2 = 0.0

    for k, v in flat.items():
        if "RouterAdapter" in k:
            adapter_l2 = adapter_l2 + jnp.sum(v ** 2)

    return adapter_l2

tx = optimizer.create_optimizer(
    **c.optimizer,
    total_steps=1000,
)

opt_state = tx.init(params)

new_params = params

for step in range(2):
    loss, grads = jax.value_and_grad(loss_fn)(new_params)

    updates, opt_state = tx.update(
        grads,
        opt_state,
        new_params,
    )

    new_params = optax.apply_updates(
        new_params,
        updates,
    )

    print("step", step, "loss =", float(loss))



flat_old = flatten_dict(params, sep="/")
flat_new = flatten_dict(new_params, sep="/")

changed = []

for k in flat_old:

    delta = jnp.max(
        jnp.abs(flat_new[k] - flat_old[k])
    )

    if float(delta) > 0:
        changed.append((k, float(delta)))

adapter_changed = [
    x for x in changed
    if "RouterAdapter" in x[0]
]

non_adapter_changed = [
    x for x in changed
    if "RouterAdapter" not in x[0]
]

print("loss =", float(loss))
print("changed_count =", len(changed))
print("adapter_changed_count =", len(adapter_changed))
print("non_adapter_changed_count =", len(non_adapter_changed))

print("\nAdapter changed params:")

for k, v in adapter_changed:
    print(k, v)

print("\nNon-adapter changed params:")

for k, v in non_adapter_changed:
    print(k, v)