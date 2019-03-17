from a3code import AlexNetFeatures
from dataimport import importImages, Dataset
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
    for i in range (len(train_img)): # class 0/1 is non-cancerous, 2,3 is cancerous
        for j in range(len(train_img[i])):
            x = alexNet(train_img[i][j][0].unsqueeze(0))
            for k in range(1,len(train_img[i][j])):
                x = torch.cat((x,alexNet(train_img[i][j][k].unsqueeze(0))), dim=2)
            print (x.shape)
            filename = 'train_features' + str(i) + "_" + str(j) +'.pt'
            torch.save(x, filename)
    for i in range (len(valid_img)):
        for j in range(len(valid_img[i])):  
            x = alexNet(valid_img[i][j][0].unsqueeze(0))
            for k in range(1,len(valid_img[i][j])):
                x = torch.cat((x,alexNet(valid_img[i][j][k].unsqueeze(0))), dim=2)
 
            filename = 'valid_features' + str(i) + "_" + str(j) +'.pt'
            torch.save(x, filename)
    for i in range (len(test_img)):
        for j in range(len(test_img[i])): 
            x = alexNet(test_img[i][j][0].unsqueeze(0))
            for k in range(1,len(test_img[i][j])):
                x = torch.cat((x,alexNet(test_img[i][j][k].unsqueeze(0))), dim=2)
                

            filename = 'test_features' + str(i) + "_"+ str(j) +'.pt'
            torch.save(x, filename)

  
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv1 = nn.Conv2d(6, 12, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=2, stride = 2)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=2, stride = 2)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(4992, 2000) 
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 2)
    def forward(self, x):
        #print (x.shape)        
        x = x.permute(0, 3, 1, 2)
        #print (x.shape)
        x = F.relu(self.conv1(x))
        #print (x.shape)
        x = F.relu(self.conv2(x))
        #print (x.shape)
        x = F.relu(self.conv3(x))
        #print (x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 48 * 8 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.squeeze(1) # Flatten to [batch_size]

        return x

#get_features()

def load_features(dir): # label):
    dir = os.path.expanduser(dir)
    tensors = []
#    count = 0
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        print (d)
        tensors.append((torch.load(d).squeeze(0))) #, label))
#        if (count == 5):
#         break
#        count += 1
    return tensors

def evaluate(net, validloader, criterion):
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(validloader, 0):
        feature = data[0]#.squeeze(0)
        label = data[1]

        outputs = net(feature)
        #print (label.shape)
        loss = criterion(outputs, label.long())
        

        prediction = outputs.max(1, keepdim=True)[1]

        total_err += prediction.ne(label.long().view_as(prediction)).sum().item()
        total_loss += loss.item()
        total_epoch += len(label)

    err = float(total_err)/total_epoch
    loss = float(total_loss)/total_epoch
    return err, loss
    

def train_net(net, trainloader, valid, learning_rate=0.001, weight_decay = 0.01, num_epochs=10):
    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        #shuffle(train)
        total_train_loss = 0.0
        total_train_err = 0.0

        total_epoch = 0
        for i, data in enumerate(trainloader, 0):
        #for i in range(len(train)):
            optimizer.zero_grad()
 
            feature = data[0]#.squeeze(0)
            label = data[1]


            #print (feature.shape, label.shape)
            #target = Variable(torch.Tensor(label).long())
            #print (label.shape)
            #outputs = torch.zeros(70, 1, 2)
            prediction = 0

            outputs = net(feature)
            #print (label.shape)
            loss = criterion(outputs, label.long())
            loss.backward()
            optimizer.step()

##            
##            lossOutput = torch.zeros(1,2)
##            for j in range(len(feature)):
##                outputs[j] = (net(feature[j]))
##                if (outputs.max(1, keepdim=True) == 1):
##                    prediction = 1
##                    lossOutput = outputs[j]
##            if (prediction != 1):
##                lossOutput = torch.mean(outputs, dim=0, keepdim=True)

##            loss = criterion(lossOutput.squeeze(0), label)
##            loss.backward()
##            optimizer.step()

            prediction = outputs.max(1, keepdim=True)[1]
            #print (prediction.item(), label.item(), prediction.item() != label.item())
            #total_train_err += (prediction.item() != label.long().item())
            total_train_err += prediction.ne(label.long().view_as(prediction)).sum().item()

            total_train_loss +=loss.item()
            total_epoch += len(label)

        train_err[epoch] = float(total_train_err)/total_epoch
        train_loss[epoch] = float(total_train_loss)/(i + 1)
        val_err[epoch], val_loss[epoch] = evaluate(net, valid, criterion)
        
        print(("Epoch {}: Train err: {}, Train loss: {}, Valid err: {}, Valid loss: {}").format(
            epoch + 1,
            train_err[epoch],
            train_loss[epoch],
            val_err[epoch],
            val_loss[epoch]))
        model_path = 'Model_Epoch'+ str(epoch)
        torch.save(net.state_dict(), model_path)
            
            
            
#non_cancerous = [[1,0]]
cancerous = 1
non_cancerous = 0


#print (non_cancerous.shape)
            
train = load_features('Train_Alexnet_features_cancerous') #, cancerous)
train_labels = [non_cancerous] * len(train)
lenC = len(train)
train.extend(load_features('Train_Alexnet_features_non_cancerous')) #, non_cancerous))
train_labels.extend([cancerous] * (len(train)- lenC))
train_labels = np.array(train_labels)

train = torch.stack(train)
print (train.shape)
training_set = Dataset(train, train_labels)
trainloader = torch.utils.data.DataLoader(training_set, batch_size=5, shuffle=True,num_workers=0)

valid = load_features('Valid_Alexnet_features_cancerous') #, cancerous)
valid_labels = [non_cancerous] * len(valid)
lenC = len(valid)
valid.extend(load_features('Valid_Alexnet_features_non_cancerous')) #, non_cancerous))
valid_labels.extend([cancerous] * (len(valid)- lenC))
valid_labels = np.array(valid_labels)

valid = torch.stack(valid)
valid_set = Dataset(valid, valid_labels)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=5, shuffle=True,num_workers=0)
             
smallnet = SmallNet()
train_net(smallnet, trainloader, validloader, num_epochs=100)


