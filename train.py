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
from loaders.data_multi import PairedDataTif
import os, glob
import numpy as np
from utils.metrics_classification import ClassificationLoss, GetAUC


# import torchio as tio
# environment file

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


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

    return parser


def train(net, args, train_set, eval_set, loss_function, metrics):
    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                             pin_memory=True)

    train_loader.__code__ = ''

    # preload
    if args.preload:
        tini = time.time()
        print('Preloading...')
        train_filenames = []
        eval_filenames = []
        for i, x in enumerate(tqdm(train_loader)):
            train_filenames.append(x['filenames'])
            pass
        for i, x in enumerate(tqdm(eval_loader)):
            eval_filenames.append(x['filenames'])
            pass
        print('Preloading time: ' + str(time.time() - tini))
        train_filenames = [y for x in train_filenames for y in x]
        eval_filenames = [y for x in eval_filenames for y in x]
        # make sure the train_filenames and eval_filenames are different
        # Flatten the lists of filenames
        train_filenames_flat = [item for sublist in train_filenames for item in sublist]
        eval_filenames_flat = [item for sublist in eval_filenames for item in sublist]
        # Check if there is no intersection between the flattened lists
        assert len(set(train_filenames_flat).intersection(set(eval_filenames_flat))) == 0

        print('Asserted that train and eval filenames are different')

    # freezing parameters
    if args.freeze:
        net.par_freeze = [y for x in [list(x.parameters()) for x in [getattr(net, 'features')]] for y in x]
    else:
        net.par_freeze = []

    """ cuda """
    if args.legacy:
        net = net.cuda()
        net = nn.DataParallel(net)

    """ training class """
    models = args.scheme
    model = getattr(__import__('engine.' + models), models).LitModel

    model = model(args=args,
                  train_loader=train_loader,
                  eval_loader=eval_loader,
                  net=net,
                  loss_function=loss_function,
                  metrics=metrics)

    """ vanilla pytorch mode"""
    if args.legacy:
        # Use pytorch without lightning
        model.overall_loop()
    else:
        # Use pytorch lightning for training, you can ignore it
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.environ.get('LOGS')) + '/checkpoints/' + args.prj + '/',
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            verbose=False,
            monitor='val_loss',
            mode='min'
        )

        # we can use loggers (from TensorBoard) to monitor the progress of training
        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(os.environ.get('LOGS'), args.prj), default_hp_metric=False)
        trainer = pl.Trainer(gpus=-1, strategy='ddp',
                             max_epochs=args.epochs, logger=tb_logger,
                             accumulate_grad_batches=args.batch_update,
                             callbacks=[checkpoint_callback],
                             auto_lr_find=True)

        # if lr == 0  run learning rate finder
        if args.lr == 0:
            trainer.tune(model, train_loader, eval_loader)
        else:
            trainer.fit(model, train_loader, eval_loader)


def split_N_fold(L, fold, split):
    N = L // fold
    split10 = [list(range(i * N, (i + 1) * N)) for i in range(fold)]
    split10[-1] = list(range(split10[-1][0], L))
    eval_index = split10.pop(int(split))
    train_index= [y for x in split10 for y in x]
    return train_index, eval_index


def split_moaks(x, split):
    moaks_id = x.loc[(~x['READPRJ'].isna())]['ID'].unique()
    eval_index = [y // 2 for y in (x.loc[(~x['ID'].isin(moaks_id)) & (x['SIDE'] == 'LEFT')]).index.values]
    train_index_all = [y // 2 for y in (x.loc[(x['ID'].isin(moaks_id)) & (x['SIDE'] == 'LEFT')]).index.values]
    N = len(train_index_all) // 5
    if split == '0':
        train_index = train_index_all[N:]
    if split == '1':
        train_index = train_index_all[:N] + train_index_all[2 * N:]
    if split == '2':
        train_index = train_index_all[:2 * N] + train_index_all[3 * N:]
    if split == '3':
        train_index = train_index_all[:3 * N] + train_index_all[4 * N:]
    if split == '4':
        train_index = train_index_all[:4 * N]
    return train_index, eval_index


if __name__ == "__main__":
    from dotenv import load_dotenv
    import argparse

    parser = args_train()

    # additional arguments for dataset
    # Data
    parser.add_argument('--env', type=str, default=None, help='environment_to_use')
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

    # Model-specific Arguments

    models = parser.parse_known_args()[0].scheme
    model = getattr(__import__('engine.' + models), models).LitModel
    try:
        parser = model.add_model_specific_args(parser)
        print('Model specific arguments added')
    except:
        print('No model specific arguments')

    args = parser.parse_args()

    if args.env is not None:
        load_dotenv('env/.' + args.env)
    else:
        load_dotenv('env/.t09b')

    # Label
    from utils.get_labels import get_labels
    train_labels, val_labels, full_labels, full_subjects, train_subjects, val_subjects = get_labels()

    # Dataset
    train_set = PairedDataTif(root=os.environ.get('DATASET') + args.dataset + '/train/',
                             path='a_b', labels=train_labels, crop=args.cropsize, mode='test')
    print(len(train_set))

    eval_set = PairedDataTif(root=os.environ.get('DATASET') + args.dataset + '/val/',
                             path='a_b', labels=val_labels, crop=args.cropsize, mode='test')
    print(len(eval_set))

    # Networks
    if args.scheme == 'siamese':
        from models.MRPretrainedSiamese import MRPretrainedSiamese
        net = MRPretrainedSiamese(args_m=args)
    else:
        from models.MRPretrained import MRPretrained
        net = MRPretrained(args_m=args)

    loss_function = ClassificationLoss()
    metrics = GetAUC()

    os.makedirs(os.path.join(os.environ.get('LOGS'), args.prj, 'checkpoints'), exist_ok=True)

    args.not_tracking_hparams = ['mode', 'port', 'parallel', 'epochs', 'legacy']

    o = train_set.__getitem__(100)
    #print((o['filenames'][0]), o['filenames'][-1])
    o = eval_set.__getitem__(0)
    #print((o['filenames'][0]), o['filenames'][-1])

    train(net, args, train_set, eval_set, loss_function, metrics)

    #  USAGE
    # python train.py --backbone alexnet --fuse max2 --direction a_b --scheme siamese

