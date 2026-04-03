import torch
from torch.utils.data import Dataset
from models_and_co.utils import standardize_per_channel

class TUHEEGHealthy_NPZ_Dataset(Dataset):
    def __init__(self, X, ages=None, normalize=True, supervised=False, dtype=torch.float32):
        self.X, self.ages, self.normalize, self.supervised, self.dtype = X, ages, normalize, supervised, dtype

        if self.supervised:
            if self.ages is None:
                raise ValueError('ages must be provided when supervised=True')
            if len(self.X) != len(self.ages):
                raise ValueError('X and ages must have the same length')

    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx):
        epoch = self.X[idx]
        
        if epoch.ndim != 2:
            raise ValueError(f"expected epoch shape (C, T), got {epoch.shape}")

        x = torch.from_numpy(epoch).to(dtype=self.dtype)

        if self.normalize:
            x = standardize_per_channel(x)
        
        if self.supervised:
            y = torch.as_tensor(self.ages[idx], dtype=self.dtype)
            return x, y

        return x
