import pandas as pd
import os, glob
import numpy as np


def get_labels():
    x = pd.read_csv('env/womac4_moaks.csv')

    painR = x.loc[x['SIDE'] == 'RIGHT', ['ID', 'VER', 'SIDE', 'V$$WOMKP#']]
    painL = x.loc[x['SIDE'] == 'LEFT', ['ID', 'VER', 'SIDE', 'V$$WOMKP#']]

    pain = pd.merge(painR, painL, how='outer', on=['ID', 'VER'])
    pain['labels'] = pain['V$$WOMKP#_x'] > pain['V$$WOMKP#_y']
    pain['subjects'] = [str(x) + '_' + str(y).zfill(2) for x, y in zip(pain['ID'].values, pain['VER'].values)]

    train_subjects = [x.split('/')[-1][:-8] for x in
                      sorted(glob.glob('/home/ghc/Dataset/paired_images/womac4/' + '/train/a/*'))]
    train_subjects = list(set(train_subjects))
    val_subjects = [x.split('/')[-1][:-8] for x in
                    sorted(glob.glob('/home/ghc/Dataset/paired_images/womac4/' + '/val/a/*'))]
    val_subjects = list(set(val_subjects))

    train_labels = pain.loc[pain['subjects'].isin(train_subjects), :]['labels'].values.astype(np.uint8)
    val_labels = pain.loc[pain['subjects'].isin(val_subjects), :]['labels'].values.astype(np.uint8)
    return train_labels, val_labels
