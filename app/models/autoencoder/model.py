# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:03:28 2023

@author: r.osipovskiy
"""
import pytorch_lightning as pl
from pytorch_lightning import callbacks, cli_lightning_logo, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
    
import torch
import torch.nn.functional as F
from torch import nn


class LitAutoEncoder(LightningModule):
    """
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, 
                 input_features: int, 
                 hidden_dim = None,
                 bottleneck = None,
                 lr: float = 1e-3):
        
        super().__init__()

        hidden_dim = hidden_dim if isinstance(hidden_dim, int) else input_features // 2
        bottleneck = bottleneck if isinstance(bottleneck, int) else hidden_dim // 2
        self.layers_size = {'input_features': input_features,
                            'hidden_dim': hidden_dim,
                            'botlneck': bottleneck}
        self.lr = lr
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, bottleneck))
        
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, input_features))
        
        self.example_input_array = torch.randn(1, input_features)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _prepare_batch(self, batch):
        x, _ = batch
        # check dim 
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x)) 
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss 
