from model import Convolutional
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import numpy as np
import re
import sys, math

CLASSES = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED', 'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']
dirname = os.path.dirname(__file__)

# Display of prediction is kinda broken right now
def DisplayResults(results, images, labels):
    # check for results folder
    if not os.path.isdir(dirname + '/data/results'):
        os.mkdir(dirname + '/data/results')

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
        if(CLASSES[predicted_label] == labels[i]):
            color = 'blue'
        else:
            color = 'red'
        image_plot.set_xlabel(f"Predicted: {CLASSES[predicted_label]} ({predictions[predicted_label]:>0.1f}%) \n Actual: {labels[i]}", color=color)
    
        clrs = ['grey' if (x < max(predictions)) else 'red' for x in predictions ]
        graph_plot.barh(range(22), predictions, color=clrs)
        graph_plot.set_yticks(range(22))
        graph_plot.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        graph_plot.set_xlim([0, 100])
        graph_plot.set_yticklabels(CLASSES)

        fig.tight_layout()
        fig.savefig(dirname + '/data/results/' + str(i) + '_' + labels[i] + '.png')

    plt.show()

def PrintResults(results, images, filenames):
    # check for output folder
    if not os.path.isdir(dirname + '/data/output'):
        os.mkdir(dirname + '/data/output')

    correct = 0
    for i, result in enumerate(results):
        result = nnf.softmax(result, dim=0)
        result = result.detach().numpy()

        prediction = result[np.argmax(result)]
        filename = filenames[i]
        label = CLASSES[np.argmax(result)]

        additional_guesses = list()
        if prediction * 100 < 50:
            threshold = prediction - (prediction * 0.5)
            for p in result:
                if  p >= threshold and p <= prediction and p != prediction:
                    local_label = CLASSES[result.tolist().index(p)]
                    additional_guesses.append(f'{local_label} ({p * 100:.1f}%)')
        
        

        if label in filename.upper():
            correct += 1
            print('\033[92m' + f'Perdicted that {filename} is {label} ({prediction*100:.1f}%) {"or" if len(additional_guesses) > 0 else ""} {" or ".join(additional_guesses)}' + '\033[0m')
        else:
            print('\033[91m' +  f'Perdicted that {filename} is {label} ({prediction*100:.1f}%) {"or" if len(additional_guesses) > 0 else ""} {" or ".join(additional_guesses)}' + '\033[0m')
        
        
        output = Image.fromarray(images[i]).convert('L')
        
        output.save(dirname + '/data/output/' + str(i) + '-' + label + f'({prediction*100:.1f}%)' + '.png')

    print(f'{correct} out of {len(results)} are correct ({correct/len(results)*100:<.1f}%)')

def LoadImages(path):
    num_files = len(os.listdir(path))
    image_size = 100
    (images, labels) = (np.zeros((num_files, image_size, image_size)), list()) # np array
    for i, filename in enumerate(os.listdir(path)):
        image = Image.open(path+ "\\" + filename).convert('L') # Opens the file as a Pillow image
        new_image = Image.new(image.mode, (100, 100), 255)
        x, y = int((100/2)) - int(image.width/2), int(100) - int(image.height) 
        new_image.paste(image, (x,y))
        np_image = np.array(new_image) # Converts the pil image into a numpy array

        # Reformat filename
        filename = re.sub(r'\d+', '', filename)
        filename = re.sub(r'[()]', '', filename)[:-4]

        # Set
        labels.append(filename)
        images[i] = np_image
    
    return (images, labels, os.listdir(path))

def main(argv):
    # default values
    input_path = dirname + '/data/input'

    model_name = 'sigmoid+'
    model_path = f'{dirname}/models/{model_name}/{model_name}.model'
    
    # Load model
    print("Loading model", model_path)
    model = Convolutional(100)
    print(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load images
    (images, labels, filenames) = LoadImages(input_path)

    # Use model
    print("Predicting...")
    x = torch.from_numpy(images).float()
    results = model(x.unsqueeze(1))
    print("Finished predicting..")
    
    # Check if there is a folder for data
    if not os.path.isdir('MachineLearning\\NeuralNetwork\\data'):
        os.mkdir(dirname + '/data')

    # Prints out all inputs with the strongest prediction
    print("Labeled images will be saved to the data/results folder")
    PrintResults(results, images, filenames)

    # Prints put all the inputs in a graph visualization
    #print("Graphs will be saved to the data/results folder")
    #DisplayResults(results, images, labels)

if __name__ == "__main__":
   main(sys.argv[1:])