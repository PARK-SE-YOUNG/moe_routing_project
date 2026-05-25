import torch


def calculate_accuracy(outputs, targets):

    """
    Compute classification accuracy.
    """

    predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == targets).sum().item()

    accuracy = correct / targets.size(0)

    return accuracy