import functions.function as f
from models.model import Linear, Convolutional
from PIL import Image

import torch
import torch.nn.functional as nnf

import numpy as np


model = Linear()
model.load_state_dict(torch.load('trained_models/default.model'))

model.eval()

# Load images
path = "inputdata/"
classes = ['alef', 'ayin', 'bet', 'dalet', 'gimel', 'het', 'he', 'kaf', 'lamed', 'mem', 'nun', 'pe', 'qof', 'resh', 'samekh', 'shin', 'tav', 'tet', 'tsadi', 'vav', 'yod', 'zayin']
#letters = ['ז', 'י', 'ו', 'צ', 'ט', 'ת', 'ש', 'ס', 'ר', 'ק', 'פ', 'נ', 'מ', 'ל', 'כ', 'ה', 'ח', 'ג', 'ד', 'ב', 'ע', 'א']

(images, labels, names, num_files) = f.load_images(path)

x = torch.from_numpy(images).float()
results = model(x.unsqueeze(1))

#results = nnf.softmax(results[1], dim=0)
#results = results.detach().numpy()
#f.display_results_wo_graph(num_files, results, classes, images, labels)

for i, result in enumerate(results):
    percentages = nnf.softmax(result, dim=0)
    result = result.detach().numpy()
    top_guess = np.argmax(result)

    filename = names[i]
    label = classes[top_guess]
    percentage = percentages[top_guess] * 100
    
    if label in filename:
        print('\033[92m' + f'Perdicted that {filename} is {label} ({percentage}%)' + '\033[0m')
    else:
        print('\033[91m' +  f'Perdicted that {filename} is {label} ({percentage}%)' + '\033[0m')
    
    output = Image.fromarray(images[i]).convert('L')
    output.save('outputdata/' + str(i) + '-' + label + f'({percentage}%)' + '.png')
