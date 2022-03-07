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

# It is important that the model used is trained on the same structure as this one
class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        # Convolutional layers and Max pooling with activation functions
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        # Fully connected layer with activation functions
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
