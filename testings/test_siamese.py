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


def perform_eval(net, classifier, eval_set, combine=None, irange=None):
    all_label = []
    all_out = []
    all_features = []
    if irange == None:
        irange = range(len(eval_set))
    for i in tqdm(range(len(irange))):
        batch = eval_set.__getitem__(i)

        imgs = batch['img']
        labels = batch['labels']

        imgs = [x.unsqueeze(0) for x in imgs]
        imgs = [x.repeat(1, 3, 1, 1, 1).cuda() for x in imgs]

        imgs = [x - x.min() for x in imgs]
        imgs = [x / x.max() for x in imgs]

        if combine is not None: # combine imgs[0] and imgs[2]
            imgs[0] = combine_function(imgs, combine)#torch.cat([imgs[0][:12, :, ::], imgs[2][12:, :, ::]], 0)

        # inverse
        if labels == 1:
            imgs = [imgs[1], imgs[0]]

        # OLD MODEL
        #output, features = net(imgs)
        #print(features[0].shape)

        # NEW MODEL
        f0 = net(imgs[0])[1]
        f1 = net(imgs[1])[1]
        output = classifier((f0 - f1)[:, :, 0, 0])
        features = [f0.squeeze().unsqueeze(0), f1.squeeze().unsqueeze(0)]

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
    #parser.add_argument('--host', type=str, default='dummy')
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


    ## OLD MODEL
    # net = torch.load('/media/ExtHDD01/logscls/alexmax2/20.pth', map_location='cpu').cuda()
    #ckpt_path = '/media/ExtHDD01/logscls/'
    #ckpt = sorted(glob.glob(ckpt_path + 'siamese/vgg19max2/checkpoints/*.pth'))[12 - 2]
    #net = torch.load(ckpt, map_location='cpu').cuda().eval()


    def plot_prob_comparison(probX, probY, title=''):
        X_order = np.argsort(probX[:, 0])
        plt.scatter(np.linspace(0, 1, probX.shape[0]), probX[X_order, 0]);  # plt.show()
        plt.scatter(np.linspace(0, 1, probY.shape[0]), probY[X_order, 0]);
        plt.xlim(0, 1);
        plt.ylim(0, 1)
        if title == None:
            title = ''
        plt.title(title + '  mean P:' + str((probX[:,0]).float().mean().numpy())[:5] + '>' + str((probY[:,0]).float().mean().numpy())[:5])
        plt.xlabel('N');
        plt.ylabel('Probability');


    def combine_function(imgs0, combine='z0'):
        #x = torch.cat([x[:, :, :, :, 12:], y[:, :, :, :, :12]], 4)
        (a, b, addpm, aseg, aeff) = imgs0

        # masking
        #mask = (((a - addpm) > 0)).float()# * (aeff > 0)).float()
        #a = a * (1 - (mask > 0) / 1) + addpm * ((mask > 0) / 1)

        if combine == 'full':
            mask = torch.ones_like(a)

        elif combine.startswith('z'):
            mask = torch.zeros_like(a)
            if combine == 'z0':
                mask[:, :, :, :, :8] = 1
            elif combine == 'z1':
                mask[:, :, :, :, 8:16] = 1
            elif combine == 'z2':
                mask[:, :, :, :, 16:] = 1

        a = a * (1 - mask) + addpm * mask

        return a

    def main(eval_set, irange, prj, epoch):
        net = torch.load('/media/ExtHDD01/logscls/' + prj + '/checkpoints/' + epoch + '_net.pth',
                         map_location='cpu').cuda()
        classifier = torch.load('/media/ExtHDD01/logscls/' + prj + '/checkpoints/' + epoch + '_classifier.pth',
                                map_location='cpu').cuda()
        # a_b
        all_out, featuresR_L = perform_eval(net, classifier, eval_set, irange=irange)
        probR_L = torch.nn.Softmax(dim=1)(all_out)#.numpy()
        print('AUC a vs b: ' + str(metrics(val_labels, all_out)))
        (probA_B, featuresA_B) = (flip_by_label(x, val_labels) for x in (probR_L, featuresR_L)) # pro


        #fig = plt.figure()
        fig=plt.figure(figsize=(8, 6))
        plt.subplot(221)
        #plot_prob_comparison(probA_B, probAddpm_B, title='full')

        for plot, combine in zip([221, 222, 223, 224], ['full', 'z0', 'z1', 'z2']):

            plt.subplot(plot)

            all_out, featuresR_L = perform_eval(net, classifier, eval_set, irange=irange, combine=combine)
            probR_L = torch.nn.Softmax(dim=1)(all_out)  # .numpy()
            (probMix, _) = (flip_by_label(x, val_labels) for x in (probR_L, featuresR_L))
            plot_prob_comparison(probA_B, probMix, title=combine)

        fig.suptitle(prj.replace('/', ' ') + '  epoch: ' + epoch)
        #plt.show()
        plt.savefig('outimg/' + prj.replace('/', ' ') + '  epoch: ' + epoch + '.png')


    # selected index
    irange = range(0, 500, 5)
    val_labels = val_labels[irange]
    eval_set = PairedDataTif(root=root,
                             path='a_b_addpm_aseg_aeffphi0', labels=val_labels, crop=args.cropsize, mode='test')

    # Model
    root = 'local_probability/'
    for epoch in [10, 20, 30]:
        #(prj, epoch) = (root + '100_diff_cls_nopt', '10')
        (prj, epoch) = (root + '100_diff_cls', str(epoch))
        #(prj, epoch) = (root + 100_diff_cls_nopt', '10')

        main(eval_set, irange, prj, epoch)