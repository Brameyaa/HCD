from dataimport import importImages, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
import os.path
import time
from plot_graph import plot_training_curve
from random import shuffle

import pytorchUnet
from pytorchUnet import UNet


#in theory, this should work with the rest of the code, replacing alexnet
def get_unetfeatures():

    print("start")
    num_class = 2
    uNetmodel = UNet(num_class) #Make an AlexNetFeatures Object


    uNetmodel.eval() #Sets Alexnet nn class to evaluation mode
  
    train_img, train_label, valid_img, valid_label, test_img, test_label = importImages()
    print(train_img[0][0][0].shape)

#   This works
#    x = []
#    x.append(uNetmodel(train_img[0][0][0].unsqueeze(0)).squeeze(0))
#    x = torch.stack(x)
#    print(x.shape)
#    torch.save(x, 'utrain_feature_0.pt')
    
    for i in range (len(train_img)):
        for j in range(len(train_img[i])):
            x = []
            for k in range(len(train_img[i][j])):
                x.append(uNetmodel(train_img[i][j][k].unsqueeze(0)).squeeze(0))
            x = torch.stack(x)
            filename = 'utrain_features' + str(i) + "_" + str(j) +'.pt'
            print(x.shape)
            torch.save(x, filename)
            
    for i in range (len(valid_img)):
        for j in range(len(valid_img[i])):  
            x = []
            for k in range(len(valid_img[i][j])):
                x.append(uNetmodel(valid_img[i][j][k].unsqueeze(0)).squeeze(0))
            x = torch.stack(x)
            filename = 'uvalid_features' + str(i) + "_" + str(j) +'.pt'
            torch.save(x, filename)
    for i in range (len(test_img)):
        for j in range(len(test_img[i])): 
            x = []
            for k in range(len(test_img[i][j])):
                x.append(uNetmodel(test_img[i][j][k].unsqueeze(0)).squeeze(0))
            x = torch.stack(x)
            filename = 'utest_features' + str(i) + "_"+ str(j) +'.pt'
            torch.save(x, filename)

#get_unetfeatures()
