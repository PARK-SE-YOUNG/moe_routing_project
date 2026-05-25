import torch
from src.utils.metrics import calculate_accuracy


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        criterion,
        optimizer,
        logger,
        device="cpu"
    ):
    

        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.model.to(self.device)

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0

        for batch_idx, (x, y) in enumerate(self.dataloader):

            x = x.to(self.device)
            y = y.to(self.device)

            # Forward
            outputs = self.model(x)

            # Loss
            loss = self.criterion(outputs, y)
            accuracy = calculate_accuracy(outputs, y)

            # Backward
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy

            self.logger.info(
                f"Batch [{batch_idx+1}/{len(self.dataloader)}] "
                f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
            )

        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = total_accuracy / len(self.dataloader)

        return avg_loss, avg_accuracy
    