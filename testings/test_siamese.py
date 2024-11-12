import time, os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import os, glob
import numpy as np
from train import args_train
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrics_classification import GetAUC

metrics = GetAUC()

# NEW DATALOADER
from loaders.data_multi import PairedDataTif

# import torchio as tio
# environment file

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def plot_tsne():
    pp0 = prob0
    pp1 = prob1
    aa = np.argsort(pp0[:, 0])
    plt.scatter(np.linspace(0, 1, pp1.shape[0]), pp1[aa, 0]);  # plt.show()
    plt.scatter(np.linspace(0, 1, pp0.shape[0]), pp0[aa, 0]);
    plt.xlim(0, 1);
    plt.ylim(0, 1);
    plt.show()
    plt.scatter(pp0[:, 0], pp1[:, 0]);
    plt.xlim(0, 1);
    plt.ylim(0, 1);
    plt.show()


def quick_save_2d_to_3d(source, destination):
    #source = '/home/ghc/Dataset/paired_images/womac4/val/b/'
    #destination = '/media/ghc/Ghc_data3/OAI_diffusion_final/diffusion_classification/b/'
    source = '/home/ghc/Dataset/paired_images/womac4/val/a2d/'
    destination = '/home/ghc/Dataset/paired_images/womac4/val/a/'
    os.makedirs(destination, exist_ok=True)
    xlist = sorted(glob.glob(source + '/*'))

    subjects = sorted(list(set([x.rsplit("_", 1)[0].split('/')[-1] for x in xlist])))
    for s in subjects[:]:
        print(s)
        slices = sorted(glob.glob(source + '/' + s + '*'))
        npy = np.stack([tiff.imread(x) for x in slices], axis=0)
        if npy.shape[0] == 23:
            tiff.imwrite(destination + '/' + s + '.tif', npy)


def perform_eval(eval_set):
    all_label = []
    all_out = []
    all_features = []
    for i in tqdm(range(len(eval_set))):
        batch = eval_set.__getitem__(i)

        imgs = batch['img']
        labels = batch['labels']

        imgs = [x.unsqueeze(0) for x in imgs]
        imgs = [x.repeat(1, 3, 1, 1, 1).cuda() for x in imgs]

        imgs = [x - x.min() for x in imgs]
        imgs = [x / x.max() for x in imgs]

        # inverse
        if labels == 1:
            imgs = [imgs[1], imgs[0]]

        output, features = net(imgs)
        features = [x.cpu().detach() for x in features]
        features = torch.stack(features, dim=2)

        # metrics
        all_label.append(labels)
        all_out.append(output.cpu().detach())
        all_features.append(features)

    all_label = np.stack(all_label, 0)
    all_out = torch.cat(all_out, dim=0)
    all_features = torch.cat(all_features, dim=0)
    return all_out, all_features


def flip_by_label(x, labels):
    for i in range(x.shape[0]):
        # flip the first dimension
        if labels[i] == 1:
            x[i, ::] = torch.flip(x[i, ::], [0])
    return x


if __name__ == "__main__":
    from dotenv import load_dotenv
    import argparse

    parser = args_train()

    # additional arguments for dataset
    # Data
    parser.add_argument('--env', type=str, default='t09', help='environment_to_use')
    parser.add_argument('--dataset', type=str, default='womac4')
    parser.add_argument('--models', type=str, default='siamese')
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
    parser.add_argument('--scheme', type=str)
    parser.add_argument('--fcls', type=int, default=512)
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--epoch', type=str)

    # Model-specific Arguments

    models = parser.parse_known_args()[0].scheme
    models = 'siamese'
    model = getattr(__import__('engine.' + models), models).LitModel
    try:
        parser = model.add_model_specific_args(parser)
        print('Model specific arguments added')
    except:
        print('No model specific arguments')

    args = parser.parse_args()

    # Data parameters
    root = '/media/ghc/Ghc_data3/OAI_diffusion_final/diffusion_classification/'
    args.resize = 384
    args.cropsize = None

    # Label
    from utils.get_labels import get_labels
    _, _, full_labels, full_subjects, train_subjects, val_subjects = get_labels()
    train_index = [list(full_subjects).index(x) for x in train_subjects]
    val_index = [list(full_subjects).index(x) for x in val_subjects]
    train_labels = full_labels[train_index]
    val_labels = full_labels[val_index]

    # Model
    #ckpt = '/media/ExtHDD01/logscls/cat0/checkpoints/100.pth'
    net = torch.load('/media/ExtHDD01/logscls/alexmax2/20.pth', map_location='cpu').eval().cuda()

    #net = torch.load('/media/ExtHDD01/logscls/contrastive0/checkpoints/50.pth', map_location='cpu').eval().cuda()

    #net = torch.load('/media/ExtHDD01/logscls/contrastive/only_cls/checkpoints/100_net.pth', map_location='cpu').eval().cuda()

    ## OLD MODEL
    # Model
    #ckpt_path = '/media/ExtHDD01/logscls/'
    #ckpt = sorted(glob.glob(ckpt_path + 'siamese/vgg19max2/checkpoints/*.pth'))[12 - 2]
    #net = torch.load(ckpt, map_location='cpu').cuda().eval()

    # prob a_b
    eval_set = PairedDataTif(root=root,
                             path='a_b', labels=val_labels, crop=args.cropsize, mode='test')
    all_out, features0 = perform_eval(eval_set)
    prob0 = torch.nn.Softmax(dim=1)(all_out)#.numpy()
    print('AUC a vs b: ' + str(metrics(val_labels, all_out)))
    (prob0, features0) = (flip_by_label(x, val_labels) for x in (prob0, features0))

    eval_set = PairedDataTif(root=root,
                       path='addpm_b', labels=val_labels, crop=args.cropsize, mode='test')
    all_out, features1 = perform_eval(eval_set)
    prob1 = torch.nn.Softmax(dim=1)(all_out)#.numpy()
    print('AUC addpm vs a: ' + str(metrics(val_labels, all_out)))
    (prob1, features1) = (flip_by_label(x, val_labels) for x in (prob1, features1))


    # compare probability
    pp0 = prob0
    pp1 = prob1
    aa = np.argsort(pp0[:,0])
    plt.scatter(np.linspace(0, 1, pp1.shape[0]), pp1[aa, 0]);#plt.show()
    plt.scatter(np.linspace(0, 1, pp0.shape[0]), pp0[aa, 0]);plt.xlim(0,1);plt.ylim(0,1);plt.show()
    plt.scatter(pp0[:, 0], pp1[:, 0]);plt.xlim(0,1);plt.ylim(0,1);plt.show()

