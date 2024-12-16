import torch
import os, glob
import numpy as np
from train import args_train
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrics_classification import GetAUC
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc


metrics = GetAUC()

# NEW DATALOADER
from loaders.data_multi import PairedDataTif

# import torchio as tio
# environment file

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def plot_TSNE(data, legends, irange):
    tout = TSNE(n_components=2).fit_transform(np.concatenate(data, 0))

    p_item = []
    l_item = []
    for i in irange:
        if i > 0:
            L0 = np.concatenate(data[:i], 0).shape[0]
        else:
            L0 = 0
        if i < len(data) - 1:
            L1 = np.concatenate(data[:i + 1], 0).shape[0]
        else:
            L1 = tout.shape[0]
        p_item.append(plt.scatter(tout[L0:L1, 0], tout[L0:L1, 1], label=str(i), s=1))
        l_item.append(legends[i])
    plt.legend(p_item, l_item)
    plt.show()


def flip_by_label(x, labels):
    for i in range(x.shape[0]):
        # flip the first dimension
        if labels[i] == 1:
            x[i, ::] = torch.flip(x[i, ::], [0])
    return x


def swap_by_label(x: torch.Tensor, y: torch.Tensor, label: list) -> tuple:
    mask = torch.tensor(label, dtype=torch.bool, device=x.device)
    x_new, y_new = x.clone(), y.clone()
    temp = x_new[mask].clone()
    x_new[mask] = y_new[mask]
    y_new[mask] = temp
    return x_new, y_new


def use_classifier():
    clsAV = classifier((featureAV - featureBV).unsqueeze(0).unsqueeze(0).cuda()).detach().cpu().squeeze().numpy()
    clsBV = classifier((featureBV - featureAV).unsqueeze(0).unsqueeze(0).cuda()).detach().cpu().squeeze().numpy()

    auc = roc_auc_score(np.concatenate([np.ones(len(featureAV)), np.zeros(len(featureBV))]),
                        np.concatenate([clsAV, clsBV], 0)[:, 0])
    print(auc)


def SVM_classification(xTrain, xVal, yTrain, yVal):
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain.numpy())
    xVal = scaler.transform(xVal.numpy())

    svm = SVC(kernel='rbf', gamma='auto', probability=True)
    svm.fit(xTrain, yTrain)

    pred = svm.predict(xVal)
    pred_prob = svm.predict_proba(xVal)
    acc = accuracy_score(yVal, pred)
    # AUC
    fpr, tpr, _ = roc_curve(yVal, pred_prob[:, 1])
    auc_score = auc(fpr, tpr)
    print(f"SVM accuracy: {acc:.4f}")
    print(f"SVM AUC: {auc_score:.4f}")

    return pred_prob


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
        feature = projector(feature)

        all_feature.append(feature.detach().cpu())

    all_feature = torch.cat(all_feature, dim=0)
    return all_feature


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
    parser.add_argument('--epoch', type=str)

    # Model-specific Arguments
    args = parser.parse_args()

    # Data parameters
    root = '/home/ghc/Dataset/paired_images/womac4/'  # path to your validation data
    args.resize = 384
    args.cropsize = 0

    # Label
    from utils.get_labels import get_labels
    _, _, full_labels, full_subjects, train_subjects, val_subjects = get_labels()
    train_index = [list(full_subjects).index(x) for x in train_subjects]
    val_index = [list(full_subjects).index(x) for x in val_subjects]
    train_labels = full_labels[train_index]
    val_labels = full_labels[val_index]

    # Model
    (prj_name, epoch) = ('contrastive/111_run2_lr2/', 80)
    #(prj_name, epoch) = ('contrastive/only_diff_cls', 40)

    net = torch.load('/media/ExtHDD01/logscls/' + prj_name + '/checkpoints/' + str(epoch) + '_net.pth', map_location='cpu').eval().cuda()
    classifier = torch.load('/media/ExtHDD01/logscls/' + prj_name + '/checkpoints/' + str(epoch) + '_classifier.pth', map_location='cpu').eval().cuda()
    projector = torch.load('/media/ExtHDD01/logscls/' + prj_name + '/checkpoints/' + str(epoch) + '_projector.pth', map_location='cpu').eval().cuda()

    # checking train a vs b
    train_a = PairedDataTif(root=root + 'train/',
                             path='a', labels=train_labels, crop=args.cropsize, mode='test')
    train_b = PairedDataTif(root=root + 'train/',
                             path='b', labels=train_labels, crop=args.cropsize, mode='test')
    featureAT = perform_eval(train_a)  # features train A = proj(backbone(A))
    featureBT = perform_eval(train_b)  # features train B = proj(backbone(B))

    # checking eval a vs b
    eval_a = PairedDataTif(root=root + 'val/',
                             path='a', labels=val_labels, crop=args.cropsize, mode='test')
    eval_b = PairedDataTif(root=root + 'val/',
                             path='b', labels=val_labels, crop=args.cropsize, mode='test')

    featureAV = perform_eval(eval_a)
    featureBV = perform_eval(eval_b)

    # addpm
    eval_addpm = PairedDataTif(root=root + 'val/',
                             path='addpm', labels=val_labels, crop=args.cropsize, mode='test')
    featureAVddpm = perform_eval(eval_addpm)

    data = [featureAV, featureBV, featureAVddpm][:]
    data = [x.squeeze().numpy() for x in data]
    plot_TSNE(data, legends=['AV', 'BV', 'AVddpm'], irange=[0, 2])


    #  SVM (R - L)
    featureTR, featureTL = swap_by_label(featureAT, featureBT, train_labels)
    feature_train = featureTR - featureTL
    featureVR, featureVL = swap_by_label(featureAV, featureBV, val_labels)
    feature_val = featureVR - featureVL

    data = [featureVR, featureVL][:2]
    data = [x.squeeze().numpy() for x in data]
    plot_TSNE(data)

    prob_VRL = SVM_classification(xTrain=feature_train, xVal=feature_val, yTrain=train_labels, yVal=val_labels)
    probVAB, _ = swap_by_label(torch.from_numpy(prob_VRL[:, 0]), torch.from_numpy(prob_VRL[:, 1]), val_labels)

    # SVM (A, B)
    feature_train = torch.cat([featureAT, featureBT], 0)
    feature_val = torch.cat([featureAV, featureBV], 0)
    prob_AB = SVM_classification(xTrain=feature_train, xVal=feature_val,
                                 yTrain=torch.cat([torch.ones(len(featureAT)), torch.zeros(len(featureBT))], 0),
                                 yVal=torch.cat([torch.ones(len(featureAV)), torch.zeros(len(featureBV))], 0))

    # SVM (A, Addpm)
    feature_train = torch.cat([featureAT, featureBT], 0)
    feature_val = torch.cat([featureAV, featureAVddpm], 0)
    prob_AddpmB = SVM_classification(xTrain=feature_train, xVal=feature_val,
                                 yTrain=torch.cat([torch.ones(len(featureAT)), torch.zeros(len(featureBT))], 0),
                                 yVal=torch.cat([torch.ones(len(featureAV)), torch.zeros(len(featureBV))], 0))

    # SWAP
    if 0:
        featureVR, featureVL = swap_by_label(featureAV, featureAVddpm, val_labels)
        feature_val = featureVR - featureVL
        prob_VRL = SVM_classification(xTrain=featureTR - featureTL, xVal=feature_val, yTrain=train_labels, yVal=val_labels)
        probVAAddpm, _ = swap_by_label(torch.from_numpy(prob_VRL[:, 0]), torch.from_numpy(prob_VRL[:, 1]), val_labels)


        featureVR, featureVL = swap_by_label(featureAVddpm, featureBV, val_labels)
        feature_val = featureVR - featureVL
        prob_VRL = SVM_classification(xTrain=featureTR - featureTL, xVal=feature_val, yTrain=train_labels, yVal=val_labels)
        probVAddpmB, _ = swap_by_label(torch.from_numpy(prob_VRL[:, 0]), torch.from_numpy(prob_VRL[:, 1]), val_labels)



        ABsort = np.argsort(probVAB)
        plt.scatter(np.linspace(0, len(probVAB), len(probVAB)), probVAB[ABsort])
        plt.scatter(np.linspace(0, len(probVAB), len(probVAB)), probVAAddpm[ABsort])
        plt.show()
