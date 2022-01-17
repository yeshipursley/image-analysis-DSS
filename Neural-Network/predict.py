import torch
from torch import nn
import math

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

from model import TestNet

model = TestNet()
model.load_state_dict(torch.load('number.model'))
model.eval()

# Load images
path = 'images/'
num_files = len(os.listdir(path))
images = np.zeros((num_files, 28, 28)) # np array
for i, filename in enumerate(os.listdir(path)):
    pil_image = Image.open(path + filename).convert('L') # Opens the file as a Pillow image
    np_image = np.array(pil_image) # Converts the pil image into a numpy array
    images[i] = np_image


def predict(image_array):
    predictions = list()
    with torch.no_grad():
        for image in image_array:
            x = torch.from_numpy(image).float()

            x = (x.unsqueeze(0).unsqueeze(0))

            prediction = model(x)
            value = np.argmax(prediction.numpy())
            predictions.append(value)
        
    return predictions

    
predictions = predict(images)

num_cols = 3
num_rows = math.ceil(num_files / num_cols)
plt.figure(figsize=(2 * num_cols, 2 * 2 * num_rows))
for i in range(num_files):
    plt.subplot(num_rows, 2 * num_cols, 2*i+1)
    plt.imshow(images[i])
    plt.axis('off')
    plt.title(f"Predicted: {predictions[i]}")
plt.tight_layout()
plt.show()