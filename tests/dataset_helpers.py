import torch
from torch.utils.data import Dataset


class Generic_Dataset(Dataset):
    def __init__(self, data):

        # First Dimension (N) needs to be the samples
        # Second Dimension (D) is the dimensionality of the data
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]
