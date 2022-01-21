from weakref import ref
import torch
from torch import nn
import torch.nn.functional as nnf

import math
import re

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

from model import HebrewNet
from model import ConvolutionalNet

model = HebrewNet()
model.load_state_dict(torch.load('number.model'))
model.eval()

classes = ['alef', 'het', 'mem', 'shin']

# Load images
path = 'images/'
num_files = len(os.listdir(path))
(images, labels) = (np.zeros((num_files, 50, 50)), list()) # np array
for i, filename in enumerate(os.listdir(path)):
    pil_image = Image.open(path + filename).convert('L') # Opens the file as a Pillow image
    pil_image = pil_image.resize((50, 50))
    np_image = np.array(pil_image) # Converts the pil image into a numpy array

    # Reformat filename
    filename = re.sub(r'\d+', '', filename)
    filename = re.sub(r'[()]', '', filename)[:-4]

    # Set
    labels.append(filename)
    images[i] = np_image


def predict(image_array):
    predictions = list()
    with torch.no_grad():
        for image in image_array:
            x = torch.from_numpy(image).float()
            x = (x.unsqueeze(0).unsqueeze(0))

            prediction = model(x)
            perc = np.array(nnf.softmax(prediction[0], dim=0))
            predictions.append(perc)
        
    return predictions

    
results = predict(images)

fig,ax=plt.subplots(num_files,2)
fig.title = "Predictions"

for i, (image_plot, graph_plot) in enumerate(ax):
    predictions = results[i] * 100
    print(predictions)
    predicted_label = np.argmax(predictions)

    image_plot.imshow(images[i], cmap='gray')
    image_plot.set_xticks([])
    image_plot.set_yticks([])
    image_plot.set_xlabel(f"Predicted: {classes[predicted_label]} ({predictions[predicted_label]:>0.1f}%) \n Actual: {labels[i]}")

    clrs = ['grey' if (x < max(predictions)) else 'red' for x in predictions ]
    graph_plot.barh(range(4), predictions, color=clrs)
    graph_plot.set_yticks(range(4))
    graph_plot.set_yticklabels(classes)

fig.set_figheight(10)
plt.tight_layout()
plt.show()