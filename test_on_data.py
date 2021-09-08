%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import shutil
import pickle
from torchvision import datasets, utils
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from random import shuffle
from IPython.display import Image
from PIL import Image
import os
from steganography import Steganography

##test on data using the steganography method
os.chdir(os.getcwd()+'/data_to_be_test/data_to_be_test')
img1=Image.open('kodim03.png')
img2=Image.open('kodim24.png')
merged_image=Steganography.merge(img1,img2)
unmerged=Steganography.unmerge(merged_image)
#show the result
img2
img1
merged_image
unmerged





##test using nn


TEST_PATH=os.path.abspath(os.path.join(os.getcwd(),'..'))#set the path of the data to be tested.

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
        pin_memory=True, shuffle=True, drop_last=True)#generate test_loader

def customized_loss(S_prime, C_prime, S, C, B):
    ''' Calculates loss specified on the paper.'''
    loss_cover = (torch.nn.functional.mse_loss(C_prime, C))
    loss_secret = (torch.nn.functional.mse_loss(S_prime, S))
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret

def imshow(img, idx, learning_rate, beta):
    '''Prints out an image given in tensor format.'''
    
    img = denormalize(img, std, mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example '+str(idx)+', lr='+str(learning_rate)+', B='+str(beta))
    plt.show()
    return

def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image

for idx, test_batch in enumerate(test_loader):##plot the testing result
     # Saves images
    test_losses=[]
     
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
    
    #if idx in [1,2,3,4]:#
    print ('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.data.item(), loss_secret.data.item(), loss_cover.data.item()))

    # Creates img tensor
    imgs = [test_secret.data, test_output.data, test_cover.data, test_hidden.data]
    imgs_tsor = torch.cat(imgs, 0)

    # Prints Images
    imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)
        
    test_losses.append(test_loss.data.item())
    

    
mean_test_loss = np.mean(test_losses)

print ('Average loss on test set: {:.2f}'.format(mean_test_loss))#calculate the loss
