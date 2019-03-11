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
   for target in sorted(os.listdir(dir)):
     d = os.path.join(dir, target)
     filename, ext = os.path.splitext(os.path.basename(d))
     filename = filename.split('_')
     img = io.imread(d) #plt.imread(d)[:, :, :3]
     x = transform(img)
     label = filename[2]
     x = x.permute(2,0,1) 
     #x = x.reshape([3, 224, 224])
     images.append(x)
     labels.append(label)
   return images, np.array(labels)
   

  def importImages ():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_img1, train_labels1 = make_dataset('Training_data/Normal', 0, '.tif', transform)
    train_img2, train_labels2 = make_dataset('Training_data/Benign', 0, '.tif', transform)
    train_img3, train_labels3 = make_dataset('Training_data/In Situ', 1, '.tif', transform)
    train_img4, train_labels4 = make_dataset('Training_data/Invasive', 1, '.tif', transform)
    
    train_img = []
    train_label = []
    train_img.extend([train_img1, train_img2, train_img3, train_img4])
    train_label.extend([train_label1, train_label2, train_label3, train_label4])
    
    
    return train_img, train_label
    
  
