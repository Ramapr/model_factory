# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:23:26 2023

@author: r.osipovskiy
"""

import numpy as np 
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
#from torchvision import transforms

#%%
class dset(Dataset):
    """
    dataset for Feed Forward AE model 
    """
    def __init__(self, X, transform=None, unsqz=False):
        self.array = torch.unsqueeze(torch.from_numpy(X).float(), dim=1) if unsqz else torch.from_numpy(X).float()
        # m = array.mean(0, keepdim=True)
        # s = array.std(0, unbiased=False, keepdim=True)
        # print('finite s', torch.sum(torch.isfinite(s)))
        # self.array = (array - m) / s
        # # self.transform = transform
        # #print(torch.all(torch.isfinite( self.array ), True))


    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, idx):       
        arr = self.array[idx, :]
        # if self.transform: # normalize 2 mean=0 std=1
        #      arr = self.transform(arr)
        # print(type(arr))
        return arr, arr 


#%%

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data: np.ndarray, 
                 ratio: float = .75, 
                 batch_size : int = 32, 
                 workers: int = 2):
        
        super().__init__()
        self.data = data
        self.workers = workers
        self.ratio = ratio
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full = dset(self.data)
            # load here 
            train_size = int(len(full) * self.ratio)
            val_size = len(full) - int(len(full) * self.ratio)
            # find optimal length 
            self.train_set, self.val_set = random_split(full, [train_size, val_size])
            
    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.workers, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, num_workers=self.workers, batch_size=self.batch_size)
   
    
#%%    
    
    
    
