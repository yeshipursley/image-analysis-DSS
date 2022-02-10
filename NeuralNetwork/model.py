import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ['alef', 'ayin', 'bet', 'dalet', 'gimel', 'het', 'he', 'kaf', 'lamed', 'mem', 'nun', 'pe', 'qof', 'resh', 'samekh', 'shin', 'tav', 'tet', 'tsadi', 'vav', 'yod', 'zayin']
CLASSES_COUNT = len(CLASSES)

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.flatten = nn.Flatten()
        # Layers
        self.fc1 = nn.Linear(64*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, CLASSES_COUNT)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(torch.relu(x))
        x = self.dropout(x)
        x = self.fc3(torch.sigmoid(x))
        return x


class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(CLASSES))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
