# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:23:26 2023

@author: r.osipovskiy
"""

from torch.utils.data import Dataset
import torch
# import pytorch_lightning as pl
# from torch.utils.data import random_split, DataLoader


class dset(Dataset):
    """
    dataset for Feed Forward AE model
    """
    def __init__(self, X, unsqz=False):
        self.array = torch.unsqueeze(torch.from_numpy(X).float(), dim=1) if unsqz else torch.from_numpy(X).float()

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        arr = self.array[idx, :]
        return arr, arr
