import jax
import jax.numpy as jnp
import optax
import flax.traverse_util

from vmoe.train import trainer
from vmoe.train import optimizer
from vmoe.train import train_state as train_state_lib
from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_router_adapter_smoke
    as cfg
)
from vmoe.nn.vit_moe import VisionTransformerMoe

# --------------------------------------------------
# Config / model
# --------------------------------------------------

c = cfg.get_config()

model = VisionTransformerMoe(**c.model)

# --------------------------------------------------
# Dummy batch
# --------------------------------------------------

batch_size = 8

images = jnp.ones((batch_size, 384, 384, 3), jnp.float32)

labels = jax.nn.one_hot(
    jnp.arange(batch_size) % 1000,
    1000,
)

# --------------------------------------------------
# Init model
# --------------------------------------------------

rngs = {
    "params": jax.random.PRNGKey(0),
    "gating": jax.random.PRNGKey(1),
    "dropout": jax.random.PRNGKey(2),
}

variables = model.init(rngs, images)

params = variables["params"]

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

#tx = optax.adam(1e-3)

tx = optimizer.create_optimizer(
    **c.optimizer,
    total_steps=1000,
)

state = train_state_lib.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    rngs={
        "gating": jax.random.PRNGKey(3),
        "dropout": jax.random.PRNGKey(4),
    },
)

# --------------------------------------------------
# Save params before update
# --------------------------------------------------

params_before = flax.traverse_util.flatten_dict(
    params,
    sep="/",
)

# --------------------------------------------------
# Loss fn
# --------------------------------------------------

loss_fn = optax.softmax_cross_entropy

# --------------------------------------------------
# Run trainer.train_step()
# --------------------------------------------------

new_state = state

for step in range(2):
    new_state, metrics = trainer.train_step(
        state=new_state,
        images=images,
        labels=labels,
        loss_fn=loss_fn,
    )

    print("step", step)
    print("total_loss =", float(metrics["total_loss"]))
    print("global_norm/updates =", float(metrics["global_norm/updates"]))
# --------------------------------------------------
# Compare params
# --------------------------------------------------

#params_after = flax.traverse_util.flatten_dict(
#    new_state.params,
#    sep="/",
#)

params_after = flax.traverse_util.flatten_dict(new_state.params, sep="/")

adapter_changed = []
non_adapter_changed = []

for k in params_before:

    before = params_before[k]
    after = params_after[k]

    diff = jnp.max(jnp.abs(before - after))

    if diff > 0:

        if "RouterAdapter" in k:
            adapter_changed.append((k, float(diff)))
        else:
            non_adapter_changed.append((k, float(diff)))

# --------------------------------------------------
# Print metrics
# --------------------------------------------------

print("\n=== METRICS ===")

for k, v in metrics.items():

    if hasattr(v, "shape") and v.shape == ():
        print(k, "=", float(v))

print("\n=== PARAM UPDATE CHECK ===")

print("adapter_changed_count =", len(adapter_changed))
print("non_adapter_changed_count =", len(non_adapter_changed))

print("\nAdapter changed params:")

for k, v in adapter_changed:
    print(k, v)

print("\nNon-adapter changed params:")

for k, v in non_adapter_changed[:20]:
    print(k, v)