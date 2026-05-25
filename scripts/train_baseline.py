import yaml
import torch

from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.dummy_dataset import DummyDataset

from src.models.transformer import DenseTransformer

from src.training.losses import get_loss_function
from src.training.trainer import Trainer

from src.utils.logger import setup_logger
from src.utils.checkpoint import save_checkpoint


def load_config(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():

    # Load config
    config = load_config("configs/baseline.yaml")

    # Logger
    logger = setup_logger()

    # Set seed
    set_seed(config["seed"])

    # Dataset
    dataset = DummyDataset(
        num_samples=100,
        seq_len=config["data"]["seq_len"],
        vocab_size=config["data"]["vocab_size"],
        num_classes=config["model"]["num_classes"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # Model
    model = DenseTransformer(
        vocab_size=config["data"]["vocab_size"],
        hidden_dim=config["model"]["d_model"],
        num_heads=config["model"]["nhead"],
        num_classes=config["model"]["num_classes"]
    )

    # Loss
    criterion = get_loss_function()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        device="cpu"
    )

    # Training loop
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        logger.info(f"===== Epoch {epoch+1}/{epochs} =====")

        avg_loss, avg_accuracy = trainer.train_epoch()

        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_loss,
            filename=f"baseline_epoch_{epoch+1}.pt"
        )


if __name__ == "__main__":
    main()