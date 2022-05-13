from platform import release
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

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
            
        )

        self.fc3 = nn.Linear(128, 22)

    def forward(self, x):
        #imgs = x[0].detach().numpy()
        x = self.convolutional(x)
        #_, axs = plt.subplots(4, 4, figsize=(12, 12))
        #axs = axs.flatten()
        #imgs = x[0].detach().numpy()
        #for img, ax in zip(imgs, axs):
        #    ax.imshow(img, cmap='gray')
        #plt.show()

        x = torch.flatten(x, 1)
        x = self.fullyconnected(x)
        x = self.fc3(x)
        return x

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.flatten = nn.Flatten()
        # Layers
        self.fc1 = nn.Linear(100 * 100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 22)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(torch.relu(x))
        x = self.dropout(x)
        x = self.fc3(torch.sigmoid(x))
        return x