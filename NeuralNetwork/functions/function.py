import torch.nn.functional as nnf

import re
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os

def display_results(num_files, results, classes, images, labels):
        for i, result in enumerate(results):
            fig, ax = plt.subplots(1,2)
            image_plot = ax[0]
            graph_plot = ax[1]

            predictions = nnf.softmax(result, dim=0) * 100
            predictions = predictions.detach().numpy()
            result = result.detach().numpy()
            predicted_label = np.argmax(result)
            

            image_plot.imshow(images[i], cmap='gray')
            image_plot.set_xticks([])
            image_plot.set_yticks([])
            if(classes[predicted_label] == labels[i]):
                color = 'blue'
            else:
                color = 'red'
            image_plot.set_xlabel(f"Predicted: {classes[predicted_label]} ({predictions[predicted_label]:>0.1f}%) \n Actual: {labels[i]}", color=color)
        
            clrs = ['grey' if (x < max(predictions)) else 'red' for x in predictions ]
            graph_plot.barh(range(22), predictions, color=clrs)
            graph_plot.set_yticks(range(22))
            graph_plot.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            graph_plot.set_xlim([0, 100])
            graph_plot.set_yticklabels(classes)

            fig.tight_layout()
            fig.savefig('data/results/' + str(i) + '_' + labels[i] + '.png')

        plt.show()

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
    
    return (images, labels, os.listdir(path),num_files)
