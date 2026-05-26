import jax
import jax.numpy as jnp
import jax.experimental.mesh_utils as mesh_utils
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

from vmoe import initialization
from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012_router_adapter_smoke
    as cfg,
)
from vmoe.nn.vit_moe import VisionTransformerMoe
from vmoe.train import optimizer
from vmoe.train import train_state as train_state_lib
import ml_collections


def main():
    c = cfg.get_config()

    model = VisionTransformerMoe(**c.model)

    images = jnp.zeros((8, 384, 384, 3), jnp.float32)

    variables = model.init(
        {
            "params": jax.random.PRNGKey(0),
            "gating": jax.random.PRNGKey(1),
            "dropout": jax.random.PRNGKey(2),
        },
        images,
    )

    tx = optimizer.create_optimizer(
        **c.optimizer,
        total_steps=1000,
    )

    state = train_state_lib.TrainState.create(
        apply_fn=model.apply,
        tx=tx,
        rngs={
            "gating": jax.random.PRNGKey(3),
            "dropout": jax.random.PRNGKey(4),
        },
        **variables,
    )

    flat = flatten_dict(state.params, sep="/")

    adapter_keys = [
        k for k in flat
        if "RouterAdapter" in k
    ]

    print("num_params =", len(flat))
    print("num_adapter_params =", len(adapter_keys))
    print("checkpoint prefix =", c.initialization.prefix)
    print("initialization name =", c.initialization.name)
    print(
        "raise_if_target_unmatched =",
        c.initialization.raise_if_target_unmatched,
    )

    print("\nInitial adapter params:")
    for k in adapter_keys:
        print(k, flat[k].shape)

    devices = mesh_utils.create_device_mesh((1, 1))
    mesh = Mesh(devices, axis_names=("expert", "replica"))

    print("\nRestoring checkpoint params only...")

    params_only_rules = [
        ("head", ""),
        ("pre_logits/.*", ""),
        ("^(.*/pos_embedding)$", r"\1", "vit_zoom"),
        ("^(.*)$", r"\1"),
    ]

    restored_params = initialization.initialize_from_vmoe(
        target=state.params,
        prefix=c.initialization.prefix,
        rules=params_only_rules,
        mesh=mesh,
        axis_resources_regexes=c.initialization.axis_resources_regexes,
        raise_if_target_unmatched=c.initialization.raise_if_target_unmatched,
    )

    restored_flat = flatten_dict(restored_params, sep="/")

    restored_adapter_keys = [
        k for k in restored_flat
        if "RouterAdapter" in k
    ]

    print("\nrestore complete")
    print("num_restored_params =", len(restored_flat))
    print("num_restored_adapter_params =", len(restored_adapter_keys))

    print("\nRestored adapter params:")
    for k in restored_adapter_keys:
        print(k, restored_flat[k].shape)

    missing_adapter_keys = sorted(
        set(adapter_keys) - set(restored_adapter_keys)
    )

    if missing_adapter_keys:
        raise RuntimeError(
            f"Missing RouterAdapter params after restore: "
            f"{missing_adapter_keys}"
        )

    print("\nCheckpoint restore smoke test passed.")


if __name__ == "__main__":
    main()