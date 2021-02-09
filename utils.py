# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:16:12 2020
@author: Guillaume Godefroy
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
"""

import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
import shutil
import numpy as np


def corr2(Img1, Img2):  # correlation between the two images, metrics
    if torch.is_tensor(Img1):
        Img1 = Img1.cpu().detach().numpy()
        Img2 = Img2.cpu().detach().numpy()

    m_Img1 = np.mean(Img1)
    m_Img2 = np.mean(Img2)
    t1 = Img1 - m_Img1
    t2 = Img2 - m_Img2

    num = np.sum(np.multiply(t1, t2))
    den = np.sqrt(np.sum(t1 ** 2) * np.sum(t2 ** 2))
    r = num / den
    return r



def forced_copy(source, target):
    remove_file(target)
    shutil.copy(source, target)


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def get_lr(optimizer):
    lrs = [params['lr'] for params in optimizer.param_groups]
    return np.squeeze(np.unique(lrs))


def epoch_training(train_loader, model, criterion, metric, optimizer, epoch, n_gpus=None, print_frequency=1, regularized=False,
                   print_gpu_memory=False, vae=False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy_t = AverageMeter('Accuracy', ':.2e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, Accuracy_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        if n_gpus:
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss, batch_size, acc = batch_loss(model, images, target, criterion, metric=metric, n_gpus=n_gpus, regularized=regularized)
        if n_gpus:
            torch.cuda.empty_cache()

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)

        # compute gradient and do step
        loss.backward()
        optimizer.step()

        del loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # dice accuracy
        Accuracy_t.update(acc.item(), batch_size)

        del acc

        if i % print_frequency == 0:
            progress.display(i)

    return losses.avg, Accuracy_t.avg


def batch_loss(model, images, target, criterion, metric, n_gpus=0, regularized=False, vae=False):
    if n_gpus is not None:
        images = images.cuda()
        target = target.cuda()
        model.cuda()
    # compute output
    output = model(images)
    batch_size = images.size(0)
    if regularized:
        try:
            output, output_vae, mu, logvar = output
            loss = criterion(output, output_vae, mu, logvar, images, target)
        except ValueError:
            pred_y, pred_x = output
            loss = criterion(pred_y, pred_x, images, target)
    else:
        loss = criterion(output, target)
    output2 = torch.sigmoid(output)
    acc = metric(output2,target)
    return loss, batch_size, acc


def epoch_validatation(val_loader, model, criterion,metric, n_gpus, print_freq=1, regularized=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    Accuracy_v = AverageMeter('Accuracy', ':.2e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, Accuracy_v],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval() # batchnorm, dropout .. in eval  mod

    with torch.no_grad(): # no update of grad
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            loss, batch_size, acc = batch_loss(model, images, target, criterion, metric=metric, n_gpus=n_gpus, regularized=regularized,
                                          )

            Accuracy_v.update(acc.item(), batch_size)
            del acc
            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
            # dice accuracy


    return losses.avg, Accuracy_v.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
