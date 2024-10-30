import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
import tifffile as tiff
from torch.optim import lr_scheduler
from engine.base import BaseModel
from engine.losses import TripletCenterLoss, CenterLoss


def swap_by_label(x: torch.Tensor, y: torch.Tensor, label: list) -> tuple:
    """
    Swap elements between x and y based on label values.

    Args:
        x: Tensor of shape (B, C, H, W)
        y: Tensor of shape (B, C, H, W)
        label: List of length B with binary values

    Returns:
        tuple: (x_new, y_new) where elements are swapped based on label==1
    """
    # Convert label to boolean mask
    mask = torch.tensor(label, dtype=torch.bool)

    # Create copies to avoid modifying original data
    x_new = x.clone()
    y_new = y.clone()

    # Swap elements where label == 1
    temp = x_new[mask].clone()
    x_new[mask] = y_new[mask]
    y_new[mask] = temp

    return x_new, y_new


class LitModel(BaseModel):
    def __init__(self, args, train_loader, eval_loader, net, loss_function, metrics):
        super().__init__(args, train_loader, eval_loader, net, loss_function, metrics)

        # classification
        self.classifier = nn.Linear(self.hparams.fcls, 2)

        # contrastive learning
        self.triple = nn.TripletMarginLoss()
        if self.hparams.projection > 0:
            self.projector = nn.Linear(self.hparams.fcls, self.hparams.projection)
            self.center = CenterLoss(feat_dim=self.hparams.projection)
        else:
            self.center = CenterLoss(feat_dim=self.hparams.fcls)

        # update optimizer
        self.model_save_names = ['net', 'classifier', 'projector']
        self.para = list(self.net.parameters()) + list(self.classifier.parameters()) + list(self.projector.parameters())
        if not self.hparams.fix_center:
            self.para = self.para + list(self.center.parameters())
        self.optimizer = self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--projection', type=int, default=32)
        parser.add_argument('--fix_center', action='store_false')
        parser.add_argument('--lb_cls', type=int, default=1)
        parser.add_argument('--lb_center', type=int, default=0)
        parser.add_argument('--lb_tri', type=int, default=0)
        return parent_parser

    def training_step(self, batch, batch_idx=0):
        # training_step defined the train loop. It is independent of forward
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        if imgs[0].shape[1] == 1:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

        # get features
        _, featureA = self.net(imgs[0])
        _, featureB = self.net(imgs[1])

        # classification
        (featureR, featureL) = swap_by_label(featureA, featureB, labels)
        output = self.classifier(featureR[:, :, 0, 0] - featureL[:, :, 0, 0])
        cls_t, _ = self.loss_function(output, labels)
        self.log('cls_t', cls_t, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # contrastive loss
        if self.hparams.projection > 0:
            featureA = self.projector(featureA[:, :, 0, 0])
            featureB = self.projector(featureB[:, :, 0, 0])
        else:
            featureA = featureA[:, :, 0, 0]
            featureB = featureB[:, :, 0, 0]

        loss_t = 0
        loss_t += self.triple(featureA[:1, ::], featureA[1:, ::], featureB[:1, ::])  # (A0, A1, B0) (anchor, plus, minus)
        loss_t += self.triple(featureB[:1, ::], featureB[1:, ::], featureA[:1, ::])  # (B0, B1, A0)
        self.log('tri_t', loss_t, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        loss_center = self.center(torch.cat([f for f in [featureA, featureB]], dim=0),  # cat(A0, A1, B0, B1)
                                  torch.FloatTensor([0, 0, 1, 1]).cuda())  # cat([0, 0, 1, 1])
        self.log('ct_t', loss_center, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # total
        loss = self.hparams.lb_cls * cls_t + self.hparams.lb_tri * loss_t + self.hparams.lb_center * loss_center
        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return loss

    # @rank_zero_only
    def validation_step(self, batch, batch_idx=0):
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        if imgs[0].shape[1] == 1:
            imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
            imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)


        # get features
        _, featureA = self.net(imgs[0])
        _, featureB = self.net(imgs[1])

        # classification
        (featureR, featureL) = swap_by_label(featureA, featureB, labels)
        output = self.classifier(featureR[:, :, 0, 0] - featureL[:, :, 0, 0])
        cls_v, _ = self.loss_function(output, labels)
        self.log('cls_v', cls_v, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # contrastive loss
        if self.hparams.projection > 0:
            featureA = self.projector(featureA[:, :, 0, 0])
            featureB = self.projector(featureB[:, :, 0, 0])
        else:
            featureA = featureA[:, :, 0, 0]
            featureB = featureB[:, :, 0, 0]

        loss_t = 0
        loss_t += self.triple(featureA[:1, ::], featureA[1:, ::], featureB[:1, ::])  # (A0, A1, B0) (anchor, plus, minus)
        loss_t += self.triple(featureB[:1, ::], featureB[1:, ::], featureA[:1, ::])  # (B0, B1, A0)
        self.log('tri_v', loss_t, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        loss_center = self.center(torch.cat([f for f in [featureA, featureB]], dim=0),  # cat(A0, A1, B0, B1)
                                  torch.FloatTensor([0, 0, 1, 1]).cuda())  # cat([0, 0, 1, 1])
        self.log('ct_v', loss_center, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # total
        loss = self.hparams.lb_cls * cls_v + self.hparams.lb_tri * loss_t + self.hparams.lb_center * loss_center

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # metrics
        self.all_label.append(labels.cpu())
        self.all_out.append(output.cpu().detach())
        self.all_loss.append(loss.detach().cpu().numpy())

        return loss


#  USAGE
#  python train.py --backbone alexnet --fuse max --direction a_b --scheme contrastive0 --prj contrastive/0_012 --fcls 256 -b 2
