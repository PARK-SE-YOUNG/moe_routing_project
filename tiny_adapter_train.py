import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict

from vmoe.configs.vmoe_paper import (
    vmoe_s32_last2_ilsvrc2012_randaug_light1_ft_ilsvrc2012 as cfg
)
from vmoe.nn.vit_moe import VisionTransformerMoe
from vmoe.train import optimizer


def count_changed(old_params, new_params):
    flat_old = flatten_dict(old_params, sep="/")
    flat_new = flatten_dict(new_params, sep="/")

    changed = []
    for k in flat_old:
        delta = jnp.max(jnp.abs(flat_new[k] - flat_old[k]))
        if float(delta) > 0:
            changed.append((k, float(delta)))

    adapter_changed = [x for x in changed if "RouterAdapter" in x[0]]
    non_adapter_changed = [x for x in changed if "RouterAdapter" not in x[0]]

    return changed, adapter_changed, non_adapter_changed


def main():
    c = cfg.get_config()
    model = VisionTransformerMoe(**c.model)

    batch_size = 8
    image_size = 384
    num_classes = 1000
    num_steps = 5

    x = jnp.ones((batch_size, image_size, image_size, 3), jnp.float32)
    labels = jax.nn.one_hot(
        jnp.zeros((batch_size,), dtype=jnp.int32),
        num_classes,
    )

    rngs = {
        "params": jax.random.PRNGKey(0),
        "gating": jax.random.PRNGKey(1),
        "dropout": jax.random.PRNGKey(2),
    }

    variables = model.init(rngs, x)
    params = variables["params"]
    initial_params = params

    tx = optimizer.create_optimizer(
        **c.optimizer,
        total_steps=1000,
    )
    opt_state = tx.init(params)

    def loss_fn(params, step):
        logits, metrics = model.apply(
            {"params": params},
            x,
            rngs={
                "gating": jax.random.fold_in(jax.random.PRNGKey(3), step),
                "dropout": jax.random.fold_in(jax.random.PRNGKey(4), step),
            },
        )

        main_loss = -jnp.mean(
            jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
        )
        auxiliary_loss = metrics.get("auxiliary_loss", 0.0).mean()
        total_loss = main_loss + auxiliary_loss

        return total_loss, {
            "main_loss": main_loss,
            "auxiliary_loss": auxiliary_loss,
            "router_entropy": metrics["encoderblock_5"]["router_entropy"],
            "expert_usage_std": metrics["encoderblock_5"]["expert_usage_std"],
        }

    print("trainable_pattern =", c.optimizer.trainable_pattern)

    for step in range(num_steps):
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params, step)

        updates, opt_state = tx.update(
            grads,
            opt_state,
            params,
        )
        params = optax.apply_updates(params, updates)

        print(
            f"step={step} "
            f"loss={float(loss):.6f} "
            f"main_loss={float(metrics['main_loss']):.6f} "
            f"aux_loss={float(metrics['auxiliary_loss']):.6f} "
            f"router_entropy={float(metrics['router_entropy']):.6f} "
            f"expert_usage_std={float(metrics['expert_usage_std']):.6f}"
        )

    changed, adapter_changed, non_adapter_changed = count_changed(
        initial_params,
        params,
    )

    print()
    print("changed_count =", len(changed))
    print("adapter_changed_count =", len(adapter_changed))
    print("non_adapter_changed_count =", len(non_adapter_changed))

    print()
    print("Adapter changed params:")
    for k, v in adapter_changed:
        print(k, v)

    print()
    print("Non-adapter changed params:")
    for k, v in non_adapter_changed:
        print(k, v)

    if non_adapter_changed:
        raise RuntimeError("Non-adapter parameters changed.")

    print()
    print("Tiny adapter train loop passed.")


if __name__ == "__main__":
    main()