import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

class Classifier():
    def __init__(self, model):

        # Setup model
        self.model = Convolutional()
        self.model.load_state_dict(torch.load(model))
        self.model.eval()
    
    def Classify(self, images):
        # Convert the numpy arrays into tensors
        images = torch.from_numpy(images).float()
    
        # Fix the shape of the array
        images = images.unsqueeze(1)

        # Predict
        predictions = self.model(images)

        # Convert the predictions to a numpy array
        predictions = predictions.detach().numpy()

        # Get the most confident prediction
        topPrediction = np.argmax(predictions)

        return topPrediction

class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)


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