import numpy as np
from plot_graph import plot_training_curve

import torch

print('please place your image in \RunPrediction')
print('Ensure it is .tif format and 2048x1536 pixels')
print('please place your model in \RunPrediction')
print('Enter Model name(Model_Epoch#)')
model_path=input()

from hcd get import SmallNet

model=SmallNet()

x = torch.load(model_path)

print(x)

