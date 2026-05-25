import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import torch
from torch.utils.data import DataLoader

from src.models.transformer import SimpleTransformer
from src.datasets.dummy_dataset import DummyDataset
from src.utils.metrics import calculate_accuracy


def main():

    # Device
    device = "cpu"

    # Dataset
    dataset = DummyDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    # Model
    model = SimpleTransformer()

    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(
        "outputs/checkpoints/baseline_epoch_5.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    total_accuracy = 0.0

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            accuracy = calculate_accuracy(outputs, y)

            total_accuracy += accuracy

    avg_accuracy = total_accuracy / len(dataloader)

    print(f"\nEvaluation Accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    main()