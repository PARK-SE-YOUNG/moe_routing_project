import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    save_dir="outputs/checkpoints",
    filename="checkpoint.pt"
):

    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, checkpoint_path)

    print(f"[Checkpoint Saved] {checkpoint_path}")