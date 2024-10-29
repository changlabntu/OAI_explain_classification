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

metrics = GetAUC()
load_dotenv('.env')


def args_train():
    parser = argparse.ArgumentParser()

    # projects
    parser.add_argument('--prj', type=str, default='', help='name of the project')

    # training modes
    parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
    parser.add_argument('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    # training parameters
    parser.add_argument('-e', '--epochs', dest='epochs', default=101, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--bu', '--batch-update', dest='batch_update', default=1, type=int, help='batch to update')
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.0005, type=float, help='learning rate')

    parser.add_argument('-w', '--weight-decay', dest='weight_decay', default=0.005, type=float, help='weight decay')
    # optimizer
    parser.add_argument('--op', dest='op', default='sgd', type=str, help='type of optimizer')

    # models
    parser.add_argument('--fuse', dest='fuse', default='')
    parser.add_argument('--backbone', dest='backbone', default='vgg11')
    parser.add_argument('--pretrained', dest='pretrained', default=True)
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False)
    parser.add_argument('--classes', dest='n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--repeat', type=int, default=0, help='repeat the encoder N time')

    # misc
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')

    parser.add_argument('--dataset', type=str, default='womac4')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True)
    parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--trd', type=float, dest='trd', help='threshold of images', default=0)
    parser.add_argument('--preload', action='store_true', help='preload the data once to cache')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)

    return parser


def list_to_tensor(ax, repeat=True):
    ax = torch.cat([torch.from_numpy(x / 1).unsqueeze(2) for x in ax], 2).unsqueeze(0).unsqueeze(1)
    if repeat:
        ax = ax.repeat(1, 3, 1, 1, 1)
    ax = ax / ax.max()
    return ax


def flip_by_label(x, label):
    y = []
    for i in range(x.shape[0]):
        if label[i] == 1:
            y.append(torch.flip(x[i, :], [0]))
        else:
            y.append(x[i, :])
    return torch.stack(y, 0)


def get_xy(ax):
    if gpu:
        ax = ax.cuda()
    mask = net_gan(ax, alpha=alpha)['out0'].detach().cpu()
    mask = nn.Sigmoid()(mask)
    ax = torch.multiply(mask, ax.detach().cpu())
    return ax, mask


def gather_prob(x, flip=False):
    x = torch.cat(x, 0)
    x = nn.Softmax(dim=1)(x)
    if flip:
        x = np.array([x[i, int(labels[i] / 1)] for i in range(200)])
    else:
        x = np.array([x[i, 0] for i in range(200)])
    return x


parser = args_train()
args = parser.parse_args()

load_dotenv('env/.t09b')
x = pd.read_csv('env/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values


# Model
ckpt_path = '/media/ExtHDD01/logscls/'
ckpt = sorted(glob.glob(ckpt_path + 'siamese/vgg19max2/checkpoints/*.pth'))[12-2]
net = torch.load(ckpt, map_location='cpu')
net = net.cuda()
net = net.eval()


# diffusion data
data_root = '/home/ghc/Dataset/paired_images/womac4/full/'
alist = sorted(glob.glob(data_root + 'ap/*'))
blist = sorted(glob.glob(data_root + 'bp/*'))
clist = sorted(glob.glob(data_root + '003b/*'))


if 0:
    out_cb = []
    out_ab = []
    out_ac = []

    for i in tqdm(range(len(clist) // 23)):
        ax = [tiff.imread(x) for x in alist[i * 23:(i + 1) * 23]]
        bx = [tiff.imread(x) for x in blist[i * 23:(i + 1) * 23]]
        cx = [tiff.imread(x) for x in clist[i * 23:(i + 1) * 23]]
        ax = list_to_tensor(ax).float()
        bx = list_to_tensor(bx).float()
        cx = list_to_tensor(cx).float()

        if labels[i] == 1:
            imgs = (bx.cuda(), cx.cuda())
        else:
            imgs = (cx.cuda(), bx.cuda())
        out, (xB, xA) = net(imgs)
        out_cb.append(out.detach().cpu())

        if labels[i] == 1:
            imgs = (bx.cuda(), ax.cuda())
        else:
            imgs = (ax.cuda(), bx.cuda())
        out, (xB, xA) = net(imgs)
        out_ab.append(out.detach().cpu())

        if labels[i] == 1:
            imgs = (cx.cuda(), ax.cuda())
        else:
            imgs = (ax.cuda(), cx.cuda())
        out, (xB, xA) = net(imgs)
        out_ac.append(out.detach().cpu())


    out_ab = torch.cat(out_ab, 0)
    out_cb = torch.cat(out_cb, 0)
    out_ac = torch.cat(out_ac, 0)

    ab = nn.Softmax(dim=1)(out_ab)
    cb = nn.Softmax(dim=1)(out_cb)
    ac = nn.Softmax(dim=1)(out_ac)


    print('AUC=  ' + str(metrics(labels[:200], ab)[0]))

    ab2 = np.array([ab[i, int(labels[i] / 1)] for i in range(200)])
    cb2 = np.array([cb[i, int(labels[i] / 1)] for i in range(200)])
    ac2 = np.array([ac[i, int(labels[i] / 1)] for i in range(200)])


# GAN data
log_root = '/media/ExtHDD01/logs/womac4old/'
ckpt = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/checkpoints/net_g_model_epoch_200.pth'
net_gan = torch.load(log_root + ckpt, map_location='cpu').cuda()
alpha = 1
gpu = True


# AGAIN
out_cb = []
out_ab = []
out_ac = []
out_ac_mask = []

all_cx = []
all_mask = []
all_ax = []
all_bx = []

for i in tqdm(range(len(clist) // 23)):
    ax = [tiff.imread(x) for x in alist[i * 23:(i + 1) * 23]]
    bx = [tiff.imread(x) for x in blist[i * 23:(i + 1) * 23]]
    ax = list_to_tensor(ax, repeat=True).float()
    bx = list_to_tensor(bx, repeat=True).float()
    all_ax.append(ax)
    all_bx.append(bx)

    # GAN
    if 1: #USE GAN
        cx, mask = get_xy(ax.permute(4, 1, 2, 3, 0)[:,:1,:,:,0])
        cx = cx.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        mask = mask.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        cx_mask = torch.multiply(ax, mask)
        all_mask.append(cx_mask)
    else:
        cx = [tiff.imread(x) for x in clist[i * 23:(i + 1) * 23]]
        cx = list_to_tensor(cx, repeat=True).float()
        cx_mask = cx

    all_cx.append(cx)
    all_mask.append(cx_mask)

    imgs = (cx.cuda(), bx.cuda())
    out, (xB, xA) = net(imgs)
    out_cb.append(out.detach().cpu())

    imgs = (ax.cuda(), bx.cuda())
    out, (xB, xA) = net(imgs)
    out_ab.append(out.detach().cpu())

    imgs = (ax.cuda(), cx.cuda())
    out, (xB, xA) = net(imgs)
    out_ac.append(out.detach().cpu())

    imgs = (ax.cuda(), cx_mask.cuda())
    out, (xB, xA) = net(imgs)
    out_ac_mask.append(out.detach().cpu())

[ab3, cb3, ac3, ac3mask] = [gather_prob(x, flip=False) for x in (out_ab, out_cb, out_ac, out_ac_mask)]

plt.scatter(ac2, ac3); plt.xlim(0,1); plt.ylim(0,1); plt.show()
plt.scatter(ac3mask, ac3); plt.xlim(0,1); plt.ylim(0,1); plt.show()

#x = torch.cat(out_ab, 0)
#xx = out_ab.index_select(0, torch.LongTensor(labels[:200]))