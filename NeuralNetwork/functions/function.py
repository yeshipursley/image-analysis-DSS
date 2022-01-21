import torch
from torch import nn
import torch.nn.functional as nnf

import re
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

def display_results(num_files, results, classes, images, labels):
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
            graph_plot.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            graph_plot.set_xlim([0, 100])
            graph_plot.set_yticklabels(classes)

        fig.set_figheight(10)
        plt.tight_layout()
        plt.show()

def predict(image_array, model):
    predictions = list()
    with torch.no_grad():
        for image in image_array:
            x = torch.from_numpy(image).float()
            x = (x.unsqueeze(0).unsqueeze(0))

            prediction = model(x)
            perc = np.array(nnf.softmax(prediction[0], dim=0))
            predictions.append(perc)
        
    return predictions

def load_images(path):
    num_files = len(os.listdir(path))
    (images, labels) = (np.zeros((num_files, 32, 32)), list()) # np array
    for i, filename in enumerate(os.listdir(path)):
        pil_image = Image.open(path + filename).convert('L') # Opens the file as a Pillow image
        pil_image = pil_image.resize((32, 32))
        np_image = np.array(pil_image) # Converts the pil image into a numpy array

        # Reformat filename
        filename = re.sub(r'\d+', '', filename)
        filename = re.sub(r'[()]', '', filename)[:-4]

        # Set
        labels.append(filename)
        images[i] = np_image
    
    return (images, labels, num_files)
