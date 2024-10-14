import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
import tifffile as tiff
from torch.optim import lr_scheduler
from engine.base import BaseModel


# from pytorch_lightning.utilities import rank_zero_only

def lambda_rule(epoch):
    # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
    n_epochs_decay = 50
    n_epochs = 101
    epoch_count = 0
    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l


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

        output, features = self.net(imgs)

        loss, _ = self.loss_function(output, labels)

        if not self.args.legacy:
            self.log('train_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss

    # @rank_zero_only
    def validation_step(self, batch, batch_idx=0):
        if 1:#self.trainer.global_rank == 0:
            imgs = batch['img']
            labels = batch['labels']

            # repeat part
            if len(imgs) == 2:
                imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
                imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

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
        else:
            return 0
