import torch
import torch.nn as nn


def get_loss_function():

    return nn.CrossEntropyLoss()


def load_balancing_loss(router_probs):

    expert_usage = router_probs.mean(dim=(0, 1))

    loss = torch.mean(expert_usage ** 2)

    return loss