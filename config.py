# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:51:35 2020
@author: Guillaume Godefroy
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
"""

config = {

    "pre_training": None,
    "optimizer": "Adam",
    "loss": "MSELoss",
    "metric": "corr2",
    "n_epochs": 20,
    "save_every_n_epochs": 50,
    "initial_learning_rate": 1e-04,
    "decay_patience": 20,
    "early_stopping_patience": 20,
    "save_best": True,
    "batch_size": 32,
    "model_name": 'UneXt50',
    "regularized": None,
    "learning_rate_decay_step_size": None,
    'learning_rate_decay_patience': None
}