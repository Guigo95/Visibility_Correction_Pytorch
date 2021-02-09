# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:15:35 2020
@author: Guillaume Godefroy
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
"""

import warnings
import numpy as np
import torch
import torch.nn
import pandas as pd
from utils import epoch_training, epoch_validatation, forced_copy, get_lr

def train(model, optimizer, loss_function, n_epochs, training_loader, validation_loader,
          model_filename, metric_to_monitor="val_loss", metric=None, early_stopping_patience=None,
          learning_rate_decay_patience=None, save_best=False, n_gpus=1, verbose=True, regularized=False,
          decay_factor=0.1, min_lr=0., learning_rate_decay_step_size=None, pretrain=None):
    training_log = list()
  
    start_epoch = 0
    training_log_header = ["epoch", "loss", "lr", "val_loss",'metric_score', 'metric_val_score']

    if learning_rate_decay_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=learning_rate_decay_patience,
                                                               verbose=verbose, factor=decay_factor, min_lr=min_lr)
    elif learning_rate_decay_step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=learning_rate_decay_step_size,
                                                    gamma=decay_factor, last_epoch=-1)
        # Setting the last epoch to anything other than -1 requires the optimizer that was previously used.
        # Since I don't save the optimizer, I have to manually step the scheduler the number of epochs that have already
        # been completed. Stepping the scheduler before the optimizer raises a warning, so I have added the below
        # code to step the scheduler and catch the UserWarning that would normally be thrown.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(start_epoch):
                scheduler.step() 
    else:
        scheduler = None
        
    for epoch in range(start_epoch, n_epochs):
        
        if epoch <10 and pretrain:
             for p in model.enc0.parameters():
                 p.requires_grad = False
             for p in model.enc1.parameters():
                 p.requires_grad = False
             for p in model.enc2.parameters():
                 p.requires_grad = False
             for p in model.enc3.parameters():
                 p.requires_grad = False
             for p in model.enc4.parameters():
                 p.requires_grad = False
        else:
            for p in model.enc0.parameters():
                 p.requires_grad = True
            for p in model.enc1.parameters():
                 p.requires_grad = True
            for p in model.enc2.parameters():
                 p.requires_grad = True
            for p in model.enc3.parameters():
                 p.requires_grad = True
            for p in model.enc4.parameters():
                 p.requires_grad = True

            # early stopping
            if (training_log and early_stopping_patience
                    and np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()
                    <= len(training_log) - early_stopping_patience):
                print("Early stopping patience {} has been reached.".format(early_stopping_patience))
                break

            # train the model
            loss, metric_score = epoch_training(training_loader, model, loss_function, optimizer=optimizer, metric=metric,
                                                epoch=epoch,
                                                n_gpus=n_gpus,
                                                regularized=regularized)
            try:
                training_loader.dataset.on_epoch_end()
            except AttributeError:
                warnings.warn("'on_epoch_end' method not implemented for the {} dataset.".format(
                    type(training_loader.dataset)))

            # predict validation data
            if validation_loader:
                val_loss, metric_val_score = epoch_validatation(validation_loader, model, loss_function, metric=metric,
                                                                n_gpus=n_gpus,
                                                                regularized=regularized )
                metric_val_score = 1 - metric_val_score
            else:
                val_loss = None

            # update the training log
            training_log.append([epoch, loss, get_lr(optimizer), val_loss, metric_score,
                                 metric_val_score])  # each epoch add results to training log
            pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(
                model_filename + 'log.csv')

            min_epoch = np.asarray(training_log)[:, training_log_header.index(metric_to_monitor)].argmin()

            # check loss and decay
            if scheduler:
                if validation_loader and scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(val_loss)  # case plateau on validation set
                elif scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(loss)
                else:
                    scheduler.step()

            # save model
            torch.save(model.state_dict(), model_filename)
            if save_best and min_epoch == len(training_log) - 1:
                best_filename = model_filename.replace(".h5", "_best.h5")
                forced_copy(model_filename, best_filename)

