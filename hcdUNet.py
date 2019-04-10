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
from torch.utils import data
from skimage import io

class Dataset(data.Dataset):
    #This object class will be used right before training to formalize the data into an object for pytorch to use in torch.utils.data.DataLoader
   def __init__(self, image, labels):
     self.labels = labels
     self.image = image
   def __len__(self):
     return len(self.image)
   def __getitem__(self, index):
     X = self.image[index]
     y = self.labels[index]
     return X, y

def make_dataset(dir, label, extensions, transform):
   images = []
   labels = []
   
   dir = os.path.expanduser(dir)
   #count = 0

   for target in sorted(os.listdir(dir)):
     d = os.path.join(dir, target)
     filename, ext = os.path.splitext(os.path.basename(d))
     filename = filename.split('_')
     #print (dir, filename, ext)
     img = io.imread(d) #plt.imread(d)[:, :, :3]
     img = img.astype(np.float32)
     #x = transform(img)
     x = torch.from_numpy(img) #transform(img)
     label = label
     #print (filename, label)
     #x = x.permute(2,0,1)
     images.append(x)
     labels.append(label)
     #count += 1
     #if count == 5:
     #   break

   return images, np.array(labels)
   

def importImages ():
    #Prepares datasets using images in foler and the make_dataset function defined above
    #Label '1' means cancerous tissue
    #Label '0' means non cancerous tissue
    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_img1, train_labels1 = make_dataset('Training_data_unet_feat/Normal', 0, '.tif', transform)
    train_img2, train_labels2 = make_dataset('Training_data_unet_feat/Invasive', 1, '.tif', transform)

    valid_img1, valid_labels1 = make_dataset('Validation_data_unet_feat/Normal', 0, '.tif', transform)
    #valid_img2, valid_labels2 = make_dataset('Validation_data/Benign', 0, '.tif', transform)
    #valid_img3, valid_labels3 = make_dataset('Validation_data/In Situ', 1, '.tif', transform)
    valid_img4, valid_labels4 = make_dataset('Validation_data_unet_feat/Invasive', 1, '.tif', transform)

    test_img1, test_labels1 = make_dataset('Test_data_unet_feat/Normal', 0, '.tif', transform)
    test_img2, test_labels2 = make_dataset('Test_data_unet_feat/Benign', 0, '.tif', transform)
    test_img3, test_labels3 = make_dataset('Test_data_unet_feat/In Situ', 1, '.tif', transform)
    test_img4, test_labels4 = make_dataset('Test_data_unet_feat/Invasive', 1, '.tif', transform)

    train_img = []
    train_label = []
    train_img.extend([train_img1, train_img2])
    train_label.extend([train_labels1, train_labels2])

    valid_img = []
    valid_label = []
    valid_img.extend([valid_img1, valid_img4])
    valid_label.extend([valid_labels1, valid_labels4])
    
    test_img = []
    test_label = []
    test_img.extend([test_img1, test_img2, test_img3, test_img4])
    test_label.extend([test_labels1, test_labels2, test_labels3, test_labels4])
    return train_img, train_label, valid_img, valid_label, test_img, test_label

def get_unetdata ():
    print("getting data")
    ##have already passed all the images through unet for features
    train_img, train_label, valid_img, valid_label, test_img, test_label = importImages()
    train_set = Dataset(train_img, train_label)
    valid_set = Dataset(valid_img, valid_label)
    test_set = Dataset(test_img, test_label)
    return train_set, valid_set, test_set

#Our Neural network model
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(512 * 1 * 1, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 512 * 1 * 1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]

        return x

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

def get_confidence (net, dataloader, criterion): #Evaluate the network on the validation set
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    confidence = []
    predictions_list = []
    print ("evaluate")
    for i, data in enumerate(dataloader, 0):
        img = data[0].squeeze(0)
        label = data[1]
        
        predictions = []
        prediction = 0

        for j in range(len(img)):
            outputs = net(img[j].unsqueeze(0))
            loss = criterion(outputs, label.long())
              
            predictions.append(outputs.max(1, keepdim=False)[1].item())
            #print (outputs.max(1, keepdim=False)[1].item())
        
        if (1 in predictions):
            prediction = 1
            confidence.append(len(predictions[predictions==1])/len(predictions))
        confidence.append(len(predictions[predictions==0])/len(predictions))
        predictions_list.append(prediction)
        #print (len(predictions),predictions[0], prediction)
        total_err += prediction != label #prediction.ne(label.long().view_as(prediction)).sum().item()
        total_loss += loss.item()
        total_epoch += len(label)
    
    err = float(total_err)/total_epoch
    loss = float(total_loss)/(i + 1)
    return err, loss, predictions_list, confidence

def train_net(net, trainloader, valid, learning_rate=0.00001, weight_decay = 0.0001, num_epochs=10):
    print("Training")
    #Train Function
    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    start_time = time.time()

    for epoch in range(num_epochs):
        #shuffle(train)
        total_train_loss = 0.0
        total_train_err = 0.0

        total_epoch = 0
        for i, data in enumerate(trainloader, 0):
            feature = data[0]
            label = data[1]
            prediction = 0
            #Begin Forward Pass
            outputs = net(feature)
            #print (label.shape)
            loss = criterion(outputs, label.long()) #compute total Loss

            #Begin Backward Pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # a clean up step for PyTorch

            prediction = outputs.max(1, keepdim=True)[1]  #maxpool the result

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

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    epochs = np.arange(1, num_epochs + 1)

    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
            
cancerous = 1
non_cancerous = 0

training_set, valid_set, test_set = get_unetdata()
trainloader = torch.utils.data.DataLoader(training_set, batch_size=200, shuffle=True,num_workers=0)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False,num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,num_workers=0)

print("Model Initiating")             
smallnet = SmallNet()
#state = torch.load('Model_Epoch_#')
#smallnet.load_state_dict(state)

#training
train_net(smallnet, trainloader, validloader, num_epochs=10)

#plotting - enter the train parameters and destination
print("plot")
plot_training_curve('Model_Epoch',10)

print("test vals")
test_err, test_loss = get_accuracy(smallnet, testloader, nn.CrossEntropyLoss())
print('Test err: ', test_err)
print('Test loss: ', test_loss)
test_err, test_loss, predictions_list, confidence = get_confidence(smallnet, testloader, nn.CrossEntropyLoss())

print ('Test Accuracy: ', str(1.0-test_err))
