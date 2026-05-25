import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples=100,
        seq_len=16,
        vocab_size=100,
        num_classes=2
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # Random token sequence
        x = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,)
        )

        # Random label
        y = torch.randint(
            low=0,
            high=self.num_classes,
            size=(1,)
        ).item()

        return x, y