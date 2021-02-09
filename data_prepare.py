# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:16:28 2020
@author: Guillaume Godefroy
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
"""

import torch
import numpy as np
from scipy.io.matlab import mio


class DataGenerator(torch.utils.data.Dataset):


    def __init__(self, list_IDs, path_data):

        if isinstance(list_IDs, list):
            self.list_IDs = list_IDs
        else:
            self.list_IDs = [list_IDs]

        self.path_data = path_data

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]
        data = mio.loadmat(self.path_data + ID)
        x_temp = data["X"]  # X is the name of the input in the matlab file
        y_temp = data["Yf"]  # Yf is the name of the ground truth in the matlab file

        x_temp = np.real(x_temp) - np.min(np.real(x_temp))

        x_temp = x_temp / np.max(x_temp)
        x_temp = x_temp.astype('float32')
        x_temp = np.expand_dims(x_temp, 0)

        y_temp = y_temp / np.max(y_temp)
        y_temp = y_temp.astype('float32')
        y_temp =np.expand_dims(y_temp, 0)

        y = torch.from_numpy(y_temp)
        x = torch.from_numpy(x_temp)

        return x, y


     


