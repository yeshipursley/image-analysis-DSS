import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalMain(nn.Module):
    def __init__(self):
        super(ConvolutionalMain, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

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

class HorseShoe(nn.Module):
    def __init__(self):
        super(HorseShoe, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

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

class Lines(nn.Module):
    def __init__(self):
        super(Lines, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

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

class Fours(nn.Module):
    def __init__(self):
        super(Fours, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

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
    
class Mixed(nn.Module):
    def __init__(self):
        super(Mixed, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

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