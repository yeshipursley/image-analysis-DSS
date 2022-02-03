import functions.function as f
from models.model import Linear, Convolutional
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as nnf

import numpy as np


model = Convolutional()
model.load_state_dict(torch.load('trained_models/default.model'))
model.eval()

# Load images
path = "data/input/"
classes = ['alef', 'ayin', 'bet', 'dalet', 'gimel', 'het', 'he', 'kaf', 'lamed', 'mem', 'nun', 'pe', 'qof', 'resh', 'samekh', 'shin', 'tav', 'tet', 'tsadi', 'vav', 'yod', 'zayin']

(images, labels, names, num_files) = f.load_images(path)

x = torch.from_numpy(images).float()
results = model(x.unsqueeze(1))

f.display_results(num_files, results, classes, images, labels)
for i, result in enumerate(results):
        percentages = nnf.softmax(result, dim=0)
        result = result.detach().numpy()
        top_guess = np.argmax(result)

        filename = names[i]
        label = classes[top_guess]
        percentage = percentages[top_guess] * 100
        
        if label in filename:
            print('\033[92m' + f'Perdicted that {filename} is {label} ({percentage:.1f}%)' + '\033[0m')
        else:
            print('\033[91m' +  f'Perdicted that {filename} is {label} ({percentage:.1f}%)' + '\033[0m')
        
        output = Image.fromarray(images[i]).convert('L')
        output.save('data/output/' + str(i) + '-' + label + f'({percentage:.1f}%)' + '.png')
