# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:27:48 2021

@author: 13732
"""

from itertools import islice
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import random
import shutil
import pickle
from torchvision import datasets, utils
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from random import shuffle
from IPython.display import Image

import neural_network

#Manipulating data
def sample_train(src,dst,rate):
    for sub in os.listdir(src):
        pathDir=os.listdir(os.path.join(src,sub))
        filenumber=len(pathDir)
        sample=random.sample(pathDir,int(filenumber*rate*0.8))
        for name in sample:
            shutil.copyfile(os.path.join(os.path.join(src,sub),name),os.path.join(dst,name))
    return

def sample_valid(src,dst,rate):
    for sub in os.listdir(src):
        pathDir=os.listdir(os.path.join(src,sub))
        filenumber=len(pathDir)
        sample=random.sample(pathDir,int(filenumber*rate*0.2))
        for name in sample:
            shutil.copyfile(os.path.join(os.path.join(src,sub),name),os.path.join(dst,name))
    return
src='./data'
dst_train=os.getcwd()+'/data/sample/train_folder/train'
dst_valid=os.getcwd()+'/data/sample/valid_folder/valid'

sample_train(src,dst_train,0.2)
sample_valid(src,dst_valid,0.2)


# TODO: Define train, validation and models
MODELS_PATH = os.getcwd()+'/output/models/'
# TRAIN_PATH = cwd+'/train/'
# VALID_PATH = cwd+'/valid/'
# VALID_PATH ='./data/sample/valid'
TRAIN_PATH =os.getcwd()+'/data/sample/train_folder'
TEST_PATH =os.getcwd()+'./data/sample/valid_folder'

if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)



# Creates training set

# Hyper Parameters
num_epochs = 3
batch_size = 2
learning_rate = 0.0001
beta = 1

# Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
        TRAIN_PATH,
        transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std)
        ])), batch_size=batch_size, num_workers=1, 
        pin_memory=True, shuffle=True, drop_last=True)

# Creates test set
test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
        TEST_PATH, 
        transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std)
        ])), batch_size=2, num_workers=1, 
        pin_memory=True, shuffle=True, drop_last=True)




## train the nn.model based on the sample here!!


net, mean_train_loss, loss_history =neural_network.train_model(train_loader, beta, learning_rate)


# Plot loss through epochs
plt.plot(loss_history)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()

# net.load_state_dict(torch.load(MODELS_PATH+'Epoch N4.pkl'))

# Switch to evaluate mode
net.eval()

test_losses = []
# Show images
for idx, test_batch in enumerate(test_loader):
     # Saves images
    data, _ = test_batch

    # Saves secret images and secret covers
    test_secret = data[:len(data)//2]
    test_cover = data[len(data)//2:]

    # Creates variable from secret and cover images
    test_secret = Variable(test_secret, volatile=True)
    test_cover = Variable(test_cover, volatile=True)

    # Compute output
    test_hidden, test_output = net(test_secret, test_cover)
    
    # Calculate loss
    test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta)
    
#     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))
    
#     print (diff_S, diff_C)
    
    if idx in [1,2,3,4]:
        print ('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.data.item(), loss_secret.data.item(), loss_cover.data.item()))

        # Creates img tensor
        imgs = [test_secret.data, test_output.data, test_cover.data, test_hidden.data]
        imgs_tsor = torch.cat(imgs, 0)

        # Prints Images
        imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)
        
    test_losses.append(test_loss.data.item())
        
mean_test_loss = np.mean(test_losses)

print ('Average loss on test set: {:.2f}'.format(mean_test_loss))

