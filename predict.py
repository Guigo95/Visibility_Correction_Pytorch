# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:58:58 2020
@author: Guillaume Godefroy
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
"""

import os
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn
import model
from scipy.io.matlab import mio
from config import *
from data_prepare import DataGenerator

path_data ='Ts/'

save_path = "results/"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
n_gpus = 1

model = getattr(model, config['model_name'])(pretrain=None)
model.train()
# %%
checkpoint = torch.load('model/test.h5')
model.load_state_dict(checkpoint)
name_file = os.listdir(path_data)

params1 = {
    'path_data': path_data,
}

random.shuffle(name_file)
data = {"test": name_file}

params2 = {'batch_size': 32,
           'shuffle': True
           }


test_generator = DataGenerator(data['test'], **params1)
testset = DataLoader(test_generator, **params2)
#%%
save = 1
pred_tot = []
brut_tot = []
true_tot = []
for i, (images, target) in enumerate(testset):


    if n_gpus is not None:
        brut = images
        images = images.cuda()
        target = target

        model.cuda()
        output = model(images)
        pred_tot.append(output.cpu().data.numpy())
        brut_tot.append(brut.data.numpy())
        true_tot.append(target.data.numpy())
        if save:
            tosave ={}
            name_save = 'prediction' + str(i) + '.mat'
            target = target.cpu().data.numpy()
            brut = brut.cpu().data.numpy()
            img = output.cpu().data.numpy()
            tosave['yt' ] = target
            tosave['y' ] = img
            tosave['x' ] = brut
            mio.savemat(save_path +name_save, tosave)
#%%
plt.figure(1)

img = output.cpu().detach().numpy()

handle = plt.subplot(311)
handle.set_title('row image')
plt.imshow(brut[0, 0, :, :], cmap='hot')

handle = plt.subplot(312)
handle.set_title('target image')
plt.imshow(target[0, 0, :, :], cmap='hot')

handle = plt.subplot(313)
handle.set_title('reconstructed image')
plt.imshow(img[0, 0, :, :], cmap='hot')
plt.show()
