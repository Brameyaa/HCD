from a3code import AlexNetFeatures

def get_features ():
  alexNet = AlexNetFeatures()
  alexNet.eval()
  
  
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
