from platform import release
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolutional(nn.Module):
    def __init__(self, size):
        super(Convolutional, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 6, 5), # 1 input image channel, 6 output channels, 5x5 square convolution
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # If you want to change the input size, change the variables after 16 (channels) here
            # The input needs to be an single array, so the length is width * heigth
            # To calculate the size at this stage use this formula: (input_size / 4) - 3, with result rounded down
            nn.Conv2d(6, 16, 5), # 6 input image channel, 16 output channels, 5x5 square convolution
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        size = int((size/4) - 3)
        self.fullyconnected = nn.Sequential(
            nn.Linear(16 * size * size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 22)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.fullyconnected(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(16 * 23 * 23, 240) 
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 22)

    def forward(self, x):
         # First convolution and pooling
        x = torch.sigmoid(self.conv1(x))
        x = self.pool(x)

        # Second convolution and pooling
        x = torch.sigmoid(self.conv2(x))
        x = self.pool(x)

        # Flatten dimensions
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__() # Input size is 224, 224
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96, 256, kernel_size=6, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(3,2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 22)
        )
        
    def forward(self, x):
        # Convolution layers
        x = self.conv(x)
        # Flatten
        x = torch.flatten(x,1)
        # Fully connected layers
        x = self.fc(x)
        return x
