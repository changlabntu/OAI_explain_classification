import torch
import torch.nn as nn


class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, 512))

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        print(batch_size)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        print(dist)
        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max())  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min())  # mask[i]==0: negative samples of sample i

        for x in dist_ap:
            print(x.shape)
        for x in dist_an:
            print(x.shape)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)  # normalize data by batch size
        return loss, prec


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss