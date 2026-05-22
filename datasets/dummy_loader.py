import torch
from torch.utils.data import Dataset, DataLoader

class DummyNLPDataset(Dataset):
    """로컬 디버깅용 가짜 NLP 토큰 임베딩 데이터셋"""
    def __init__(self, num_samples=100, seq_len=16, d_model=64):
        self.data = torch.randn(num_samples, seq_len, d_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_local_dataloader(batch_size=8):
    dataset = DummyNLPDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)