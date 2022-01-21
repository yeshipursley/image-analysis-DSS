from ast import arguments
import functions.function as f
import torch

from models.model import ConvolutionalNet
from models.model import HebrewNet

model_type= "Conv"

if (model_type == "Conv"):
    model = ConvolutionalNet()
    model.load_state_dict(torch.load('trained_models/convolutional.model'))
else:
    model = HebrewNet()
    model.load_state_dict(torch.load('trained_models/default.model'))

model.eval()

# Load images
path = "inputdata/"
classes = ['alef', 'het', 'mem', 'shin']

(images, labels, num_files) = f.load_images(path)

results = f.predict(images, model)
f.display_results(num_files, results, classes, images, labels)