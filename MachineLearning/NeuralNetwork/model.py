import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 22 * 22, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 22)

    def forward(self, x):
         # First convolution and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Second convolution and pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten dimensions
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Alternate reprensentation
class Convolutional2(nn.Module):
    def __init__(self):
        super(Convolutional2, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 6, 5), # 1 input image channel, 6 output channels, 5x5 square convolution
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5), # 6 input image channel, 16 output channels, 5x5 square convolution
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        self.fullyconnected = nn.Sequential(
            nn.Linear(16 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 22)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.fullyconnected(x)
        return x
