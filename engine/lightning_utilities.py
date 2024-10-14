import torch
import torch.nn as nn


def sanity_check(train_loader, net, loss_function, metrics):
    all_label = []
    all_out = []
    for i, batch in enumerate(train_loader):
        print(i)
        (_, imgs, labels) = batch
        print('imgs: ' + str(imgs.shape))
        print('labels[0]: ' + str(labels[0].shape))
        output = net(imgs)
        print('output[0]: ' + str(output[0].shape))
        loss, _ = loss_function(output, labels)

        all_label.append(labels[0].cpu())
        all_out.append(nn.Softmax(dim=1)(output[0]).cpu().detach())
        if i == 2:
            break

    all_out = torch.cat(all_out, 0)
    all_label = torch.cat(all_label, 0)
    metrics = metrics(all_label, all_out)
    print('loss: ' + str(loss))
    print('metrics: ' + str(metrics))
