from a3code import AlexNetFeatures
from dataimport import importImages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim


def get_features ():
    alexNet = AlexNetFeatures()
    alexNet.eval()
  
    train_img, train_label, valid_img, valid_label, test_img, test_label = importImages()
    print(train_img[0][0][0].shape)
    train_features = []
    valid_features = []
    test_features = []
    for i in range (len(train_img)):
        for j in range(len(train_img[i])):
            x = []
            for k in range(len(train_img[i][j])):
                x.append(alexNet(train_img[i][j][k].unsqueeze(0)))
            train_features.append(x)
    torch.save(train_features, 'train_features.pt')
    for i in range (len(valid_img)):
        for j in range(len(valid_img[i])):  
            x = []
            for k in range(len(valid_img[i][j])):
                x.append(alexNet(valid_img[i][j][k].unsqueeze(0)))
            valid_features.append(x)
    torch.save(valid_features, 'valid_features.pt')
    for i in range (len(test_img)):
        for j in range(len(test_img[i])): 
            x = []
            for k in range(len(test_img[i][j])):
                x.append(alexNet(test_img[i][j][k].reshape([1, 3, 224, 224])))
            test_features.append(x)
    torch.save(test_features, 'test_features.pt')

  
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 2 * 2, 9)
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 2 * 2)
        x = self.fc(x)
        x = x.squeeze(1) # Flatten to [batch_size]

        return x
print("START")

get_features()

print("DONE")


