from a3code import AlexNetFeatures

def get_features ():
    alexNet = AlexNetFeatures()
    alexNet.eval()
  
  train_img, train_label = importImages()
  
  train_features = []
  valid_features = []
  test_features = []
  for i in range (len(train_img)):
    x = []
    for j in range(70):
         x.append(alexNet(train_img[i][j].reshape([1, 3, 224, 224])))
    train_features.append(x)
   torch.save(train_features, 'train_features.pt')
  for i in range (len(valid_img)):
    x = []
    for j in range(70):
         x.append(alexNet(valid_img[i][j].reshape([1, 3, 224, 224])))
    valid_features.append(x)
  torch.save(valid_features, 'valid_features.pt')
  for i in range (len(test_img)):
    x = []
    for j in range(70):
         x.append(alexNet(test_img[i][j].reshape([1, 3, 224, 224])))
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
