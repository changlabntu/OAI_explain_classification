import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import copy


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def append_parameters(blocks):
    parameters = [list(x.parameters()) for x in blocks]
    all_parameters = []
    for pars in parameters:
        for par in pars:
            all_parameters.append(par)
    return all_parameters


def to_freeze(pars):
    for par in pars:
        par.requires_grad = False


def to_unfreeze(pars):
    for par in pars:
        par.requires_grad = True


class PainNet0(nn.Module):
    def __init__(self, FilNum=[64, 128, 256, 256, 512, 512]):
        super(PainNet0, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(3, FilNum[0], 7, stride=2, padding=3, bias=True),
        nn.BatchNorm2d(FilNum[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        nn.Conv2d(FilNum[0],FilNum[1], 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(FilNum[1]),
        nn.ReLU(),
        nn.Conv2d(FilNum[1],FilNum[2], 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(FilNum[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(FilNum[2],FilNum[3], 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(FilNum[3]),
        nn.ReLU(),
        nn.Conv2d(FilNum[3],FilNum[4], 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(FilNum[4]),
        nn.ReLU(),
        nn.Conv2d(FilNum[4],FilNum[5], 3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(FilNum[5]),
        nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(28),
            nn.Linear(FilNum[5], 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)


class ResnetFeatures(nn.Module):
    def __init__(self, resnet_name, pretrained, fmap_c):
        super(ResnetFeatures, self).__init__()
        self.resnet = getattr(models, resnet_name)(pretrained=pretrained)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.fmap_c = fmap_c

        to_freeze(list(self.resnet.parameters()))
        print_num_of_parameters(self.resnet)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], self.fmap_c, 8, 8)
        return x


class MRPretrained(nn.Module):
    def __init__(self, args_m):
        super(MRPretrained, self).__init__()
        self.features = self.get_encoder(args_m)
        #self.classifier = nn.Conv2d(args_m.fcls, args_m.n_classes, 1, 1, 0)
        self.classifier = nn.Linear(args_m.fcls, args_m.n_classes)
        self.fuse = args_m.fuse
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def get_encoder(self, args_m):
        if args_m.backbone == 'pain':
            features = PainNet0().features
        elif args_m.backbone.startswith('resnet'):
            features = ResnetFeatures(args_m.backbone, pretrained=args_m.pretrained, fmap_c=self.fmap_c)
        else:
            features = getattr(models, args_m.backbone)(pretrained=args_m.pretrained).features
        return features

    def get_feature(self, x0):
        B = x0.shape[0]
        x0 = x0.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x0 = x0.reshape(B * x0.shape[1], x0.shape[2], x0.shape[3], x0.shape[4])  # (B*23, 3, 224, 224)
        # features
        x0 = self.features(x0)  # (B*23, 512, 7, 7)
        x0 = x0.reshape(B, 23, x0.shape[1], x0.shape[2], x0.shape[3])
        x0 = x0.permute(0, 2, 3, 4, 1)  # (B, 512, 7, 7, 23)
        return x0

    def forward(self, x):   # (B, 3, 224, 224, 23)
        # dummies
        out = None  # output of the model
        features = None  # features we want to further analysis
        # reshape
        B = x.shape[0]
        x = x.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x = x.reshape(B * x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # (B*23, 3, 224, 224)
        # features
        x = self.features(x)  #
        # fusion
        if self.fuse == 'cat':  # concatenate across the slices
            x = torch.mean(x, dim=(2, 3))  # (B, 512, 1, 1, 23)
            x, _ = torch.max(x, 2)
            x = self.avg(x)  # (B*23, 512, 1, 1)
            x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3])  # (B, 23, 512, 1, 1)
            xcat = x.view(B, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # (B, 23*512, 1, 1)
            out = self.classifier(xcat[:, :, 0, 0])  # (Classes)
            features = xcat

        if self.fuse == 'max':  # max-pooling across the slices
            x = self.avg(x)  # (B*23, 512, 1, 1)
            x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3])  # (B, 23, 512, 1, 1)
            features, _ = torch.max(x, 1)  # (B, 512, 1, 1)
            out = self.classifier(features[:, :, 0, 0])  # (Classes)
            #out = out[:, :, 0, 0]

        if self.fuse == 'max2':  # max-pooling across the slices
            x = torch.mean(x, dim=(2, 3))  # (B, 512, 23)
            x1 = torch.mean(x1, dim=(2, 3))
            x0, _ = torch.max(x0, 2)
            x1, _ = torch.max(x1, 2)
            out = self.classifier(x0 - x1)  # (Classes)
            #out = out[:, :, 0, 0]

        return out, features


if __name__ == '__main__':
    net = ResnetFeatures('resnet101', pretrained=True, fmap_c=2048)
    net(torch.rand(1, 3, 256, 256))
