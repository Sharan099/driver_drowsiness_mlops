import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1st Convolution Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 2nd Convolution Block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 3rd Convolution Block (optional for deeper features)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # adjust based on your input image size
        self.fc2 = nn.Linear(256, 1)  # binary output
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # flatten
        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 256).to(x.device)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # binary output between 0 and 1
        return x
