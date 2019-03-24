#------Potential Method 1-------
from skimage import io
import numpy as np

#im = io.imread() #per tif image, converts to numpy array

#-----Potential Method 2--------
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import os
import os.path

#data_training = datasets.ImageFolder(root='Training_data',
#                                     transform=transforms.ToTensor())
#data_test = datasets.ImageFolder(root='Test_data',
#                                     transform=transforms.ToTensor())


from torch.utils import data

class Dataset(data.Dataset):
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
  #Image Size: 2048 x 1536 pixels
   images = []
   labels = []
   
   dir = os.path.expanduser(dir)
   count = 0

   for target in sorted(os.listdir(dir)):
     d = os.path.join(dir, target)
     filename, ext = os.path.splitext(os.path.basename(d))
     filename = filename.split('_')
     print (dir, filename, ext)
     img = io.imread(d) #plt.imread(d)[:, :, :3]
     
     img = img.astype(np.float32)
     #x = transform(img)
     x = torch.from_numpy(img) #transform(img)
     label = label
     #print (filename, label)
     x = x.permute(2,0,1)
     result = torch.zeros((3, 1568, 2240))
   
     result[:, :x.shape[1],:x.shape[2]] = x
     y = []
     for i in range(7):
        for j in range(10):
          transformed = transform(result[:, i * 224: (i + 1) * 224, j * 224: (j + 1) * 224])
          y.append(transformed)
     #x = x.reshape([3, 224, 224])
     images.append(y)
     labels.append(label)
     #count += 1
     #if count == 5:
     #   break

   return images, np.array(labels)
   

def importImages ():
    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_img1, train_labels1 = make_dataset('Training_data/Normal', 0, '.tif', transform)
    train_img2, train_labels2 = make_dataset('Training_data/Benign', 0, '.tif', transform)
    train_img3, train_labels3 = make_dataset('Training_data/In Situ', 1, '.tif', transform)
    train_img4, train_labels4 = make_dataset('Training_data/Invasive', 1, '.tif', transform)

    valid_img1, valid_labels1 = make_dataset('Validation_data/Normal', 0, '.tif', transform)
    valid_img2, valid_labels2 = make_dataset('Validation_data/Benign', 0, '.tif', transform)
    valid_img3, valid_labels3 = make_dataset('Validation_data/In Situ', 1, '.tif', transform)
    valid_img4, valid_labels4 = make_dataset('Validation_data/Invasive', 1, '.tif', transform)

    test_img1, test_labels1 = make_dataset('Test_data/Normal', 0, '.tif', transform)
    test_img2, test_labels2 = make_dataset('Test_data/Benign', 0, '.tif', transform)
    test_img3, test_labels3 = make_dataset('Test_data/In Situ', 1, '.tif', transform)
    test_img4, test_labels4 = make_dataset('Test_data/Invasive', 1, '.tif', transform)

    train_img = []
    train_label = []
    train_img.extend([train_img1, train_img2, train_img3, train_img4])
    train_label.extend([train_labels1, train_labels2, train_labels3, train_labels4])

    valid_img = []
    valid_label = []
    valid_img.extend([valid_img1, valid_img2, valid_img3, valid_img4])
    valid_label.extend([valid_labels1, valid_labels2, valid_labels3, valid_labels4])
    
    test_img = []
    test_label = []
    test_img.extend([test_img1, test_img2, test_img3, test_img4])
    test_label.extend([test_labels1, test_labels2, test_labels3, test_labels4])
    
    
    return train_img, train_label, valid_img, valid_label, test_img, test_label
    
  
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
