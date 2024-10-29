import time, os, glob
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import tifffile as tiff
from utils.metrics_classification import ClassificationLoss, GetAUC
from loaders.data_multi import MultiData as Dataset
from dotenv import load_dotenv
import argparse
from loaders.data_multi import PairedData, PairedData3D
import matplotlib.pyplot as plt


def get_xy(ax):
    if gpu:
        ax = ax.cuda()
    mask = net_gan(ax, alpha=alpha)['out0'].detach().cpu()
    mask = nn.Sigmoid()(mask)
    ax = torch.multiply(mask, ax.detach().cpu())
    return ax, mask


# GAN model
log_root = '/media/ExtHDD01/logs/womac4old/'
ckpt = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/checkpoints/net_g_model_epoch_200.pth'
net_gan = torch.load(log_root + ckpt, map_location='cpu').cuda()
alpha = 1
gpu = True


alist = sorted(glob.glob('/media/ghc/Ghc_data3/OAI_diffusion_final/diffusion_classification/a/*'))

for i in tqdm(range(len(alist))):
    ax = tiff.imread(alist[i])
    ax = ax / ax.max()
    ax = torch.from_numpy(ax).unsqueeze(1).float()
    cx, mask = get_xy(ax)
    #cx = cx.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    #mask = mask.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
    cx_mask = torch.multiply(ax, mask)
    tiff.imwrite(alist[i].replace('/a/', '/agan/'), cx.squeeze().numpy())