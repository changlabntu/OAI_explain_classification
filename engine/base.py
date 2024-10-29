import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
import tifffile as tiff
from torch.optim import lr_scheduler
import csv
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# from pytorch_lightning.utilities import rank_zero_only

def lambda_rule(epoch):
    # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
    n_epochs_decay = 50
    n_epochs = 101
    epoch_count = 0
    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l


class BaseModel(pl.LightningModule):
    def __init__(self, args, train_loader, eval_loader, net, loss_function, metrics):
        super().__init__()

        self.learning_rate = args.lr

        self.model_save_names = ['net']

        # hyperparameters
        self.args = args
        # self.hparams = args
        hparams = {x: vars(args)[x] for x in vars(args).keys() if x not in args.not_tracking_hparams}
        hparams.pop('not_tracking_hparams', None)
        self.hparams.update(hparams)
        print(self.hparams)
        self.save_hyperparameters(self.hparams)
        self.best_auc = 0

        # adding data
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # adding training components
        self.net = net
        self.loss_function = loss_function
        self.get_metrics = metrics
        self.para = list(self.net.parameters())
        self.optimizer = self.configure_optimizers()

        self.epoch = 0

        # Begin of training
        self.tini = time.time()
        self.all_label = []
        self.all_out = []
        self.all_loss = []
        self.best_loss = np.inf

    def configure_optimizers(self):
        print('Configuring optimizers...')
        para = self.para
        print(len(para))
        print('Number of parameters in configuring optimizers ', sum(p.numel() for p in para if p.requires_grad))
        optimizer = None
        scheduler = None
        if self.args.op == 'adams':
            optimizer = torch.optim.Adam(para, lr=self.learning_rate,
                                         weight_decay=self.args.weight_decay)
        elif self.args.op == 'sgd':
            #par_freeze = set(self.net.par_freeze)
            optimizer = torch.optim.SGD(para,
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.args.weight_decay)

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {'best_auc': 0})

    def training_step(self, batch, batch_idx=0):
        # training_step defined the train loop. It is independent of forward
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
        imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

        output, _ = self.net(torch.cat(imgs, 0))
        labels = torch.cat([labels, 1 - labels]).type(torch.long).cuda()

        loss, _ = self.loss_function(output, labels)

        if not self.args.legacy:
            self.log('train_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss

    # @rank_zero_only
    def validation_step(self, batch, batch_idx=0):
        imgs = batch['img']
        labels = batch['labels']

        # repeat part
        imgs[0] = imgs[0].repeat(1, 3, 1, 1, 1)
        imgs[1] = imgs[1].repeat(1, 3, 1, 1, 1)

        imgs = torch.cat(imgs, 0)
        output, _ = self.net(imgs)
        #print(labels)
        #labels = torch.cat([torch.zeros(labels.shape), torch.ones(labels.shape)]).type(torch.long).cuda()
        #print(labels)
        labels = torch.cat([labels, 1 - labels]).type(torch.long).cuda()

        loss, _ = self.loss_function(output, labels)
        if not self.args.legacy:
            self.log('val_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        # metrics
        self.all_label.append(labels.cpu())
        self.all_out.append(output.cpu().detach())
        self.all_loss.append(loss.detach().cpu().numpy())

        return loss


    def save_auc_csv(self, auc, epoch):
        auc = auc.cpu().numpy()
        auc = np.insert(auc, 0, epoch)
        with open(os.path.join(os.environ.get('LOGS'), self.args.prj, 'auc.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(auc)

    def validation_epoch_end(self, x):
        #self.train_loader.dataset.shuffle_images()
        #self.eval_loader.dataset.shuffle_images()
        if 1:#self.trainer.global_rank == 0:
            all_out = torch.cat(self.all_out, 0)
            all_label = torch.cat(self.all_label, 0)
            metrics = self.get_metrics(all_label, all_out)

            auc = torch.from_numpy(np.array(metrics)).cuda()
            if not self.args.legacy:
                for i in range(len(auc)):
                    self.log('auc' + str(i), auc[i], on_step=False, on_epoch=True, prog_bar=True, logger=True,
                             sync_dist=True)
            self.all_label = []
            self.all_out = []
            self.tini = time.time()

            if (auc[0] > self.best_auc) and (self.epoch >= 2):
                self.best_auc = auc[0]

            # saving checkpoints
            if self.epoch % 5 == 0:
                for name in self.model_save_names:
                    file_name = os.path.join(os.environ.get('LOGS'), self.args.prj, 'checkpoints',
                                             str(self.epoch) + '_' + name + '.pth')
                    print('save model at: ' + file_name)
                    torch.save(getattr(self, name), file_name)

            self.all_loss = []
            self.epoch += 1
            self.save_auc_csv(auc[0], self.epoch)
            return metrics
        else:
            return 0

    """ Legacy Pytorch Code """
    def save_best_auc(self, auc, metrics):
        if (self.all_loss < self.best_loss) and (self.epoch >= 2):
            self.best_loss = self.all_loss
            print(self.best_loss)
            if not self.args.legacy:
                self.log('best_auc', auc[0])
            file_name = os.path.join('checkpoints', self.args.prj,
                                     (str(self.epoch) + '_{:.3f}.pth').format(metrics[0]))
            torch.save(self.net, file_name)
            print('save model at: ' + file_name)


    def training_loop(self, train_loader):
        self.net.train(mode=True)
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            loss = self.training_step(batch=batch)
            loss.backward()
            epoch_loss += loss
            if i % self.args.batch_update == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return epoch_loss / i

    def eval_loop(self, eval_loader):
        self.net.train(mode=False)
        self.net.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                loss = self.validation_step(batch=batch)
                epoch_loss += loss
            metrics = self.validation_epoch_end(x=None)
            return epoch_loss / i, metrics

    def overall_loop(self):
        for epoch in range(self.args.epochs):
            tini = time.time()
            train_loss = self.training_loop(self.train_loader)
            with torch.no_grad():
                eval_loss, eval_metrics = self.eval_loop(self.eval_loader)

            print_out = {
                'Epoch: {}': [epoch],
                'Time: {:.2f} ': [time.time() - tini],
                'Train Loss: ' + '{:.4f} ': [train_loss],
                'Loss (T/V): ' + '{:.4f} ': [eval_loss],
                'Acc: ' + '{:.4f} ' * len(eval_metrics): eval_metrics,
            }
            print(' '.join(print_out.keys()).format(*[j for i in print_out.values() for j in i]))