import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
import tifffile as tiff
from torch.optim import lr_scheduler
from engine.base import BaseModel


class LitModel(BaseModel):
    def __init__(self, args, train_loader, eval_loader, net, loss_function, metrics):
        super().__init__(args, train_loader, eval_loader, net, loss_function, metrics)

    def training_step(self, batch, batch_idx=0):
        # training_step defined the train loop. It is independent of forward
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        if len(imgs) == 2:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

        if labels == 1:
            imgs = [imgs[1], imgs[0]]

        #print((imgs[0].max(), imgs[0].min(), imgs[1].max(), imgs[1].min()))

        output, features = self.net(imgs)

        loss, _ = self.loss_function(output, labels)

        if not self.args.legacy:
            self.log('train_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss

    # @rank_zero_only
    def validation_step(self, batch, batch_idx=0):
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        if len(imgs) == 2:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

        if labels == 1:
            imgs = [imgs[1], imgs[0]]

        output, features = self.net(imgs)

        loss, _ = self.loss_function(output, labels)
        if not self.args.legacy:
            self.log('val_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        # metrics
        self.all_label.append(labels.cpu())
        self.all_out.append(output.cpu().detach())
        self.all_loss.append(loss.detach().cpu().numpy())

        return loss

