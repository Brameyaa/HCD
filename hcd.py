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
import time
from plot_graph import plot_training_curve
from random import shuffle
from torch.autograd import Variable

def get_features ():



    alexNet = AlexNetFeatures() #Make an AlexNetFeatures Object


    alexNet.eval() #Sets Alexnet nn class to evaluation mode
  
    train_img, train_label, valid_img, valid_label, test_img, test_label = importImages()
    '''
    Grabs the batched train, val and test images and data using the dataimport.py ImportImages() python script
    train_image...etc. are ouputs of the importImages() function '''


    print(train_img[0][0][0].shape)
    train_features = []
    valid_features = []
    test_features = []


    '''These 3 for loops will run each set through alexnet for feature detection (Transfer Learning step)
    output of the for loops will be .pt files that are the image data after transfer learning is applied '''
    for i in range (len(train_img)): # class 0/1 is non-cancerous, 2,3 is cancerous
        for j in range(len(train_img[i])):
            x = []
            for k in range(len(train_img[i][j])):
                x.append(alexNet(train_img[i][j][k].unsqueeze(0)).squeeze(0))
            x = torch.stack(x)
            filename = 'train_features' + str(i) + "_" + str(j) +'.pt'
            print(x.shape)
            torch.save(x, filename)
    for i in range (len(valid_img)):
        for j in range(len(valid_img[i])):  
            x = []
            for k in range(len(valid_img[i][j])):
                x.append(alexNet(valid_img[i][j][k].unsqueeze(0)).squeeze(0))
            x = torch.stack(x)
            filename = 'valid_features' + str(i) + "_" + str(j) +'.pt'
            torch.save(x, filename)
    for i in range (len(test_img)):
        for j in range(len(test_img[i])): 
            x = []
            for k in range(len(test_img[i][j])):
                x.append(alexNet(test_img[i][j][k].unsqueeze(0)).squeeze(0))
            x = torch.stack(x)
            filename = 'test_features' + str(i) + "_"+ str(j) +'.pt'
            torch.save(x, filename)


#Our Neural network model
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 2 * 2, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]

        return x

'''
    load_features will grab the .pt files (images after transfer learning) in the directories
    
    and return it in tensors for training
 
'''
def load_features(dir): # label):
    dir = os.path.expanduser(dir)
    tensors = []
    #count = 0
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        print (d)
        tensors.extend(torch.load(d).squeeze(0)) #, label))
    #    if (count == 10):
    #         break
    #    count += 1
    #tensors = torch.stack(tensors)
    return tensors

def load_features_for_valid_test(dir):
    dir = os.path.expanduser(dir)
    tensors = []
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        print (d)
        tensors.append(torch.load(d).squeeze(0)) #, label))
    #tensors = torch.stack(tensors)
    return tensors

def evaluate(net, dataloader, criterion): #Evaluate the network on the validation set
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    #print ("evaluate")
    for i, data in enumerate(dataloader, 0):
        feature = data[0]#.squeeze(0)
        labels = data[1]
        outputs = net(feature)
        loss = criterion(outputs, labels.long())
        predictions = outputs.max(1, keepdim=True)[1]
        total_err += predictions.ne(labels.long().view_as(predictions)).sum().item()
        total_loss += loss.item()
        total_epoch += len(labels)
    
    err = float(total_err)/total_epoch
    loss = float(total_loss)/(i + 1)
    return err, loss

def get_accuracy (net, dataloader, criterion): #Evaluate the network on the validation set
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    #print ("evaluate")
    for i, data in enumerate(dataloader, 0):
        img = data[0].squeeze(0)
        label = data[1]
        
        predictions = []
        prediction = 0

        for j in range(len(img)): #70 subimages
            outputs = net(img[j].unsqueeze(0))
            loss = criterion(outputs, label.long())
              
            predictions.append(outputs.max(1, keepdim=False)[1].item())
            #print (outputs.max(1, keepdim=False)[1].item())
        if (1 in predictions):
            prediction = 1
        #print (len(predictions),predictions[0], prediction)
        total_err += prediction != label #prediction.ne(label.long().view_as(prediction)).sum().item()
        total_loss += loss.item()
        total_epoch += len(label)
    
    err = float(total_err)/total_epoch
    loss = float(total_loss)/(i + 1)
    return err, loss   

def train_net(net, trainloader, valid, learning_rate=0.001, weight_decay = 0.01, num_epochs=10):
    #Train Function
    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    start_time = time.time()

    #n = 0  # the number of iterations
    for epoch in range(num_epochs):
        #shuffle(train)
        total_train_loss = 0.0
        total_train_err = 0.0

        total_epoch = 0
        for i, data in enumerate(trainloader, 0):
        #for i in range(len(train)):
            #optimizer.zero_grad() #Cleanup happens before too?
            feature = data[0]#.squeeze(0)
            label = data[1]
            prediction = 0
            #print(i, feature.shape)
            #Begin Forward Pass
            outputs = net(feature)
            #print (label.shape)
            loss = criterion(outputs, label.long()) #compute total Loss

            #Begin Backward Pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # a clean up step for PyTorch #Jenn

            prediction = outputs.max(1, keepdim=True)[1]  #maxpool the result

            total_train_err += prediction.ne(label.long().view_as(prediction)).sum().item()

            total_train_loss +=loss.item()
            total_epoch += len(label)

            #Accuracy from week 3
            #iters.append(n)
            #losses.append(float(loss) / batch_size)  # compute *average* loss
            #train_acc.append(get_accuracy(model, train=True))  # compute training accuracy
            #val_acc.append(get_accuracy(model, train=False))  # compute validation accuracy
            #n += 1

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

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    epochs = np.arange(1, num_epochs + 1)

    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
            
#get_features ()

#non_cancerous = [[1,0]]
cancerous = 1
non_cancerous = 0


#print (non_cancerous.shape)

train = load_features('Train_Alexnet_features_cancerous') #, cancerous)
train_labels = [cancerous] * len(train)
lenC = len(train)

train.extend(load_features('Train_Alexnet_features_non_cancerous')) #, non_cancerous))
train_labels.extend([non_cancerous] * (len(train)- lenC))
train_labels = np.array(train_labels)

train = torch.stack(train)
print (train.shape)
print (len(train_labels))
training_set = Dataset(train, train_labels)
trainloader = torch.utils.data.DataLoader(training_set, batch_size=100, shuffle=True,num_workers=0)


valid = load_features('Valid_Alexnet_features_cancerous') #, cancerous)
valid_labels = [cancerous] * len(valid)
lenC = len(valid)
valid.extend(load_features('Valid_Alexnet_features_non_cancerous')) #, non_cancerous))
valid_labels.extend([non_cancerous] * (len(valid)- lenC))
valid_labels = np.array(valid_labels)

print (len(valid_labels), valid[0].shape)

valid = torch.stack(valid)
valid_set = Dataset(valid, valid_labels)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False,num_workers=0)

test = load_features_for_valid_test('Test_Alexnet_features_cancerous') #, cancerous)
test_labels = [cancerous] * len(test)
lenC = len(test)
test.extend(load_features_for_valid_test('Test_Alexnet_features_non_cancerous')) #, non_cancerous))
test_labels.extend([non_cancerous] * (len(test)- lenC))
test_labels = np.array(test_labels)

test = torch.stack(test)
test_set = Dataset(test, test_labels)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,num_workers=0)
             
smallnet = SmallNet()

train_net(smallnet, trainloader, validloader, num_epochs=20)

#plotting - enter the train parameters and destination
plot_training_curve('Model_Epoch19',20)

test_err, test_loss = get_accuracy(smallnet, testloader, nn.CrossEntropyLoss())

print ('Test Accuracy: ', str(1.0-test_err))
