#------Potential Method 1-------
from skimage import io
import numpy as np

im = io.imread() #per tif image, converts to numpy array

#-----Potential Method 2--------
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

data_training = datasets.ImageFolder(root='Training_data',
                                     transform=transforms.ToTensor())
data_test = datasets.ImageFolder(root='Test_data',
                                     transform=transforms.ToTensor())

