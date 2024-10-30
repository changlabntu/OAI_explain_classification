import torch
import os, glob
import numpy as np
from train import args_train
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrics_classification import GetAUC
from sklearn.manifold import TSNE

metrics = GetAUC()

# NEW DATALOADER
from loaders.data_multi import PairedDataTif

# import torchio as tio
# environment file

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def plot_TSNE(data):
    tout = TSNE(n_components=2).fit_transform(np.concatenate(data, 0))

    for i in range(len(data)):
        if i > 0:
            L0 = np.concatenate(data[:i], 0).shape[0]
        else:
            L0 = 0
        if i < len(data) - 1:
            L1 = np.concatenate(data[:i + 1], 0).shape[0]
        else:
            L1 = tout.shape[0]
        plt.scatter(tout[L0:L1, 0], tout[L0:L1, 1], label=str(i), s=1)
    #plt.xlim(0, 80)
    #plt.ylim(0, 80)
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



def flip_by_label(x, labels):
    for i in range(x.shape[0]):
        # flip the first dimension
        if labels[i] == 1:
            x[i, ::] = torch.flip(x[i, ::], [0])
    return x


if __name__ == "__main__":
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
    args = parser.parse_args()

    # Data parameters
    root = '/media/ghc/Ghc_data3/OAI_diffusion_final/diffusion_classification/'  # path to your validation data
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
    prj_name = 'contrastive/0_111/'
    epoch = 10
    net = torch.load('/media/ExtHDD01/logscls/' + prj_name + '/checkpoints/' + str(epoch) + '_net.pth', map_location='cpu').eval().cuda()
    classifier = torch.load('/media/ExtHDD01/logscls/' + prj_name + '/checkpoints/' + str(epoch) + '_classifier.pth', map_location='cpu').eval().cuda()
    projector = torch.load('/media/ExtHDD01/logscls/' + prj_name + '/checkpoints/' + str(epoch) + '_projector.pth', map_location='cpu').eval().cuda()


    def perform_eval(eval_set):
        all_feature = []

        for i in tqdm(range(len(eval_set))):
            batch = eval_set.__getitem__(i)

            imgs = batch['img']
            labels = batch['labels']

            imgs = [x.unsqueeze(0) for x in imgs]
            imgs = [x.repeat(1, 3, 1, 1, 1).cuda() for x in imgs]

            imgs = [x - x.min() for x in imgs]
            imgs = [x / x.max() for x in imgs]

            _, feature = net(imgs[0])
            feature = feature[:, :, 0, 0]
            feature = classifier(feature)

            all_feature.append(feature.detach().cpu())

        all_feature = torch.cat(all_feature, dim=0)
        return all_feature

    # checking a vs b
    eval_a = PairedDataTif(root=root,
                             path='a', labels=val_labels, crop=args.cropsize, mode='test')
    eval_b = PairedDataTif(root=root,
                             path='b', labels=val_labels, crop=args.cropsize, mode='test')
    # during model testing, the data is flipped by label, so feature0 = right knee and feature1 = left knee
    featureA = perform_eval(eval_a)   # feature0 = R, feature1 = L
    featureB = perform_eval(eval_b)  # feature0 = R, feature1 = L
    #featureAcheck = perform_eval(eval_a)  # feature0 = R, feature1 = L

    # flip right / left knee back to pain / no pain
    #(probA_B, featureA, featureB) = (flip_by_label(x, val_labels) for x in (prob0, feature0, feature1))

    data = [featureA, featureB]#, featureAcheck][:2]
    data = [x.squeeze().numpy() for x in data]

    plot_TSNE(data)
