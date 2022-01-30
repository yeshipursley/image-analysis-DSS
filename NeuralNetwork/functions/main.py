from ast import arguments
import functions.function as f
import torch
import numpy as np

from models.model import Linear, Convolutional

model = Linear()
model.load_state_dict(torch.load('trained_models/default.model'))

model.eval()

# Load images
path = "inputdata/"
classes = ['alef', 'ayin', 'bet', 'dalet', 'gimel', 'het', 'he', 'kaf', 'lamed', 'mem', 'nun', 'pe', 'qof', 'resh', 'samekh', 'shin', 'tav', 'tet', 'tsadi', 'vav', 'yod', 'zayin']
letters = ['ז']

(images, labels, names, num_files) = f.load_images(path)

x = torch.from_numpy(images).float()
results = model(x.unsqueeze(1))
results = results.detach().numpy()

f.display_results_wo_graph(num_files, results, classes, images, labels)

for result in results:
    print(classes[np.argmax(result)])

# for i, result in enumerate(results):
#     print(f'Image {names[i]} \n')
#     print(f'Actual: {labels[i]}, Predicted: {classes[np.argmax(result)]} ({result[np.argmax(result)] * 100:.1f}%)')
#     print("--------------- Percentages -------------------")
#     for x, y in enumerate(result):
#         label = classes[x]
#         spaces = [" " for k in range(10 - len(label))]
#         print(f'{label}:{"".join(spaces)}{y * 100:.0f}%')
#     print("----------------------------------------------- \n")