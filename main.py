# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:05:47 2020
@author: Guillaume Godefroy
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
"""

import os
import random
from torch.utils.data import DataLoader
import torch.nn
import utils
import model
from train import train
from data_prepare import DataGenerator
from config import *
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
path_data = 'Tr/'
n_gpus = 1


def f_optimizer(optimizer_name, model_parameters, learning_rate=config["initial_learning_rate"]):
    return getattr(torch.optim, optimizer_name)(filter(lambda p: p.requires_grad, model_parameters), lr=learning_rate)


def f_loss_function(loss_function_name, n_gpus=n_gpus):
    try:
        loss_function = getattr(utils, loss_function_name)
    except AttributeError:
        loss_function = getattr(torch.nn, loss_function_name)()
        if n_gpus > 0:
            loss_function.cuda()
    return loss_function


def f_metrics(criterion_name, n_gpus=n_gpus):
    try:
        metric = getattr(utils, criterion_name)
    except AttributeError:
        metric = getattr(torch.nn, criterion_name)()
        if n_gpus > 0:
            metric.cuda()
    return metric


# model

model = getattr(model, config['model_name'])(pretrain=config['pre_training'])

optimizer = f_optimizer(optimizer_name=config["optimizer"],
                        model_parameters=model.parameters())
loss_function = f_loss_function(config['loss'])
metric = f_metrics(config['metric'])

# data
name_file = os.listdir(path_data)
params1 = {
    'path_data': path_data,
}

random.shuffle(name_file)
data = {"train": name_file[0:len(name_file) - 31], "validation": name_file[len(name_file) - 30:len(name_file) - 1]}

params2 = {'batch_size': config['batch_size'],
           'shuffle': True
           }

training_generator = DataGenerator(data['train'], **params1)
training_set = DataLoader(training_generator, **params2)
validation_generator = DataGenerator(data['validation'], **params1)
validation_set = DataLoader(validation_generator, **params2)
#%%
# train
train(model=model, optimizer=optimizer, loss_function=loss_function, n_epochs=config["n_epochs"], verbose=bool(1),
      training_loader=training_set, validation_loader=validation_set, model_filename='model/test.h5', metric=metric,
      metric_to_monitor='metric_val_score', early_stopping_patience=config['early_stopping_patience'],
      save_best=config['save_best'], regularized=config['regularized'], pretrain = ['pretrain'],
      learning_rate_decay_step_size=config['learning_rate_decay_step_size'],
      learning_rate_decay_patience=config['learning_rate_decay_patience'])


