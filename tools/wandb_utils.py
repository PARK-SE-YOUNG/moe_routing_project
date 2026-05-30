import os

def init_wandb(default_project="vmoe-baseline"):
    try:
        import wandb
    except ImportError:
        return None

    mode = os.environ.get("WANDB_MODE", "online")

    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "yonsei2026dl10-yonsei-university"),
        project=os.environ.get("WANDB_PROJECT", default_project),
        name=os.environ.get("WANDB_RUN_NAME", None),
        tags=[
            tag.strip()
            for tag in os.environ.get("WANDB_TAGS", "").split(",")
            if tag.strip()
        ],
        mode=mode,
    )


def log_wandb(metrics, step=None):
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb():
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()