# Importing relevant Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from torch.nn import functional as F
import copy
from torch.autograd import Variable


###############################################################################
# Feature Extraction using AlexNet pretrained model

class AlexNetFeatures(nn.Module):
    '''
    Class that loads AlexNet Feature Model ('Convolution layers') with imagenet trained weights
    
    input : image tensors with dimension Lx3x224x224
    
    output : feature tensor with dimension Lx256x6x6
    
    *L - Batch size
    
    '''
    
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True) # Loads the pretrained model weights
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

    def __init__(self):
        super(AlexNetFeatures, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.load_weights() # Copies the weights to AlexNetFeatures model layers

    def forward(self, x):
        x = self.features(x)
        return x


