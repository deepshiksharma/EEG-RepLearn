import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from models_and_co.normalization import staged_mu_law


class TUHEEGHealthyAge_NPY_Dataset(Dataset):
    """
    Args:
        - csv_path (str): path to CSV having columns ['filepath', 'age', 'gender']
        - num_channels (int): number of channels
        - T (int): length of a sample
        - dtype (torch.dtype): datatype of torch tensor to return
    """
    
    def __init__(self, csv_path, num_channels=16, T=2000, dtype=torch.float32, normalize=True):
        assert os.path.isfile(csv_path), f'CSV not found: {csv_path}'

        self.df = pd.read_csv(csv_path)

        expected = {'filepath', 'age', 'gender'}
        assert expected.issubset(set(self.df.columns)), f'CSV must contain columns: {expected}'

        self.num_channels = num_channels
        self.T = T
        self.dtype = dtype
        self.normalize = normalize

        # gender mapping (only used when supervised=True)
        self._gender_map = {'male': 0, 'female': 1}

    def __len__(self): return len(self.df)

    # loading
    def _load_npy_disk(self, path):
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return np.ascontiguousarray(arr)

    # item
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        path = os.path.abspath(row['filepath'])
        if not os.path.isfile(path):
            raise FileNotFoundError(f'npy file missing: {path} (CSV index {idx})')

        arr = self._load_npy_disk(path)

        if arr.ndim != 2:
            raise ValueError(f'npy at {path} must be 2D with shape (C, T)')

        if arr.shape[0] != self.num_channels:
            raise ValueError(
                f'Expected channels-first with C={self.num_channels}, found {arr.shape}'
            )

        if arr.shape[1] != self.T:
            raise ValueError(
                f'Expected T={self.T}, found {arr.shape[1]} at {path}'
            )
        
        if self.normalize:
            arr = staged_mu_law(arr, scale=1e4)
        
        tensor = torch.from_numpy(arr).to(dtype=self.dtype)
        
        age = row['age']
        gender = row['gender']

        if pd.isna(age) or pd.isna(gender):
            raise ValueError(f'Missing age/gender at index {idx}')

        age_val = torch.tensor(float(age), dtype=torch.float32)

        gender_str = str(gender).strip().lower()
        gender_val = torch.tensor(self._gender_map[gender_str], dtype=torch.long)

        return tensor, age_val, gender_val
