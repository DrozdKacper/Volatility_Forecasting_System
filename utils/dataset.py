import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloader(X, y, batch_size=64, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(y).float()
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)