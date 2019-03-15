from a3code import AlexNetFeatures
from dataimport import importImages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import os
import os.path
from random import shuffle
from torch.autograd import Variable

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
            #train_features.append(x)
            filename = 'train_features' + str(i) + "_" + str(j) +'.pt'
            torch.save(x, filename)
    for i in range (len(valid_img)):
        for j in range(len(valid_img[i])):  
            x = []
            for k in range(len(valid_img[i][j])):
                x.append(alexNet(valid_img[i][j][k].unsqueeze(0)))
            #valid_features.append(x)
            filename = 'valid_features' + str(i) + "_" + str(j) +'.pt'
            torch.save(x, filename)
    for i in range (len(test_img)):
        for j in range(len(test_img[i])): 
            x = []
            for k in range(len(test_img[i][j])):
                x.append(alexNet(test_img[i][j][k].reshape([1, 3, 224, 224])))
                
            #test_features.append(x)\

            filename = 'test_features' + str(i) + "_"+ str(j) +'.pt'
            torch.save(x, filename)

  
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 2 * 2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 2 * 2)
        x = self.fc(x)
        x = x.squeeze(1) # Flatten to [batch_size]

        return x

#get_features()

def load_features(dir, label):
    dir = os.path.expanduser(dir)
    tensors = []
#    count = 0
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        print (d)
        tensors.append((torch.load(d), label))
#        if (count == 5):
#         break
#        count += 1
    return tensors

def evaluate(net, valid, criterion):
    total_loss = 0.0
    total_err = 0.0

    for i in range(len(valid)):
        feature,label = train[i]
        outputs = torch.zeros(70, 1, 2)
        prediction = 0
        lossOutput = torch.zeros(1,2)
        
        for j in range(len(feature)):
            outputs[j] = (net(feature[j]))
            if (outputs.max(1, keepdim=True) == 1):
                prediction = 1
                lossOutput = outputs[j]
                
        if (prediction != 1):
            lossOutput = torch.mean(outputs, dim=0, keepdim=True)
        loss = criterion(lossOutput.squeeze(0), label)
        
        total_err += prediction != label.item()
        total_loss += loss.item()

    err = float(total_err)/len(valid)
    loss = float(total_loss)/len(valid)

    return err, loss
        

    

def train_net(net, train, valid, learning_rate=0.001, weight_decay = 0, num_epochs=10):
    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        shuffle(train)
        total_train_loss = 0.0
        total_train_err = 0.0

        total_epoch = 0
        
        for i in range(len(train)):
            optimizer.zero_grad()
            feature,label = train[i]
            #print (feature[0].shape, label)
            #target = Variable(torch.Tensor(label).long())
            #print (label.shape)
            outputs = torch.zeros(70, 1, 2)
            prediction = 0
            lossOutput = torch.zeros(1,2)
            for j in range(len(feature)):
                outputs[j] = (net(feature[j]))
                if (outputs.max(1, keepdim=True) == 1):
                    prediction = 1
                    lossOutput = outputs[j]
            if (prediction != 1):
                lossOutput = torch.mean(outputs, dim=0, keepdim=True)

            loss = criterion(lossOutput.squeeze(0), label)
            loss.backward()
            optimizer.step()

            total_train_err += (prediction != label.item())
            total_train_loss +=loss.item()
            total_epoch += 1

        train_err[epoch] = float(total_train_err)/total_epoch
        train_loss[epoch] = float(total_train_loss)/(i + 1)
        val_err[epoch], val_loss[epoch] = evaluate(net, valid, criterion)
        
        print(("Epoch {}: Train err: {}, Train loss: {}").format(
            epoch + 1,
            train_err[epoch],
            train_loss[epoch],
            val_err[epoch],
            val_loss[epoch]))
        model_path = 'Model_Epoch'+ str(epoch)
        torch.save(net.state_dict(), model_path)
            
            
            
#non_cancerous = [[1,0]]
cancerous = Variable(torch.Tensor([1]).long())
non_cancerous = Variable(torch.Tensor([0]).long())


#print (non_cancerous.shape)
            
train = load_features('Train_Alexnet_features_cancerous', cancerous)
train.extend(load_features('Train_Alexnet_features_non_cancerous', non_cancerous))

valid = load_features('Valid_Alexnet_features_cancerous', cancerous)
valid.extend(load_features('Valid_Alexnet_features_non_cancerous', non_cancerous))

smallnet = SmallNet()
train_net(smallnet, train, valid)


