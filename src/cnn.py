
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5)
        
        # Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(4*49*49, 256)   # you can adjust 256 neurons
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)          # binary output


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        # no pool here
        
        x = x.view(x.size(0), -1)  # flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # use sigmoid for binary classification
        return x