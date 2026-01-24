import torch
from torch.utils.data import Dataset


class Generic_Dataset(Dataset):
    def __init__(self, data):
        # Force torch tensor (handles numpy arrays, lists, etc.)
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)

        # Ensure at least 2D: [N, D]
        if data.ndim == 1:
            data = data.unsqueeze(1)

        self.data = data

    def __len__(self):
        # Using shape avoids the "int is not callable" failure mode
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx]