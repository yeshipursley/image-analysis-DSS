from model import Linear, Convolutional
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import numpy as np
import re
import sys, getopt

CLASSES = ['ALEF', 'BET', 'GIMEL', 'DALET', 'HE', 'VAV', 'ZAYIN', 'HET', 'TET', 'YOD', 'KAF', 'LAMED', 'MEM', 'NUN', 'SAMEKH', 'AYIN', 'PE', 'TSADI', 'QOF', 'RESH', 'SHIN', 'TAV']

# Display of prediction is kinda broken right now
def DisplayResults(results, images, labels):
    # check for results folder
    if not os.path.isdir('MachineLearning/NeuralNetwork/data/results'):
        os.mkdir('MachineLearning/NeuralNetwork/data/results')

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
        fig.savefig('MachineLearning/NeuralNetwork/data/results/' + str(i) + '_' + labels[i] + '.png')

    plt.show()

def PrintResults(results, images, filenames):
    # check for output folder
    if not os.path.isdir('MachineLearning/NeuralNetwork/output'):
        os.mkdir('MachineLearning/NeuralNetwork/data/output')

    for i, result in enumerate(results):
        percentages = nnf.softmax(result, dim=0)
        result = result.detach().numpy()
        top_guess = np.argmax(result)

        filename = filenames[i]
        label = CLASSES[top_guess]
        percentage = percentages[top_guess] * 100

        additional_guesses = list()
        if percentage < 50:
            threshold = percentages[top_guess] - (percentages[top_guess] * 0.5)
            for p in percentages:
                if  p >= threshold and p <= percentages[top_guess] and p != percentages[top_guess]:
                    local_label = CLASSES[percentages.tolist().index(p)]
                    additional_guesses.append(f'{local_label} ({p * 100:.1f}%)')
        
        

        if label in filename.upper():
            print('\033[92m' + f'Perdicted that {filename} is {label} ({percentage:.1f}%) {"or" if len(additional_guesses) > 0 else ""} {" or ".join(additional_guesses)}' + '\033[0m')
        else:
            print('\033[91m' +  f'Perdicted that {filename} is {label} ({percentage:.1f}%) {"or" if len(additional_guesses) > 0 else ""} {" or ".join(additional_guesses)}' + '\033[0m')
        
        output = Image.fromarray(images[i]).convert('L')
        
        output.save('MachineLearning/NeuralNetwork/data/output/' + str(i) + '-' + label + f'({percentage:.1f}%)' + '.png')

def LoadImages(path):
    num_files = len(os.listdir(path))
    (images, labels) = (np.zeros((num_files, 64, 64)), list()) # np array
    for i, filename in enumerate(os.listdir(path)):
        pil_image = Image.open(path+ "\\" + filename).convert('L') # Opens the file as a Pillow image
        pil_image = pil_image.resize((64,64), resample=Image.NEAREST)
        np_image = np.array(pil_image) # Converts the pil image into a numpy array

        # Reformat filename
        filename = re.sub(r'\d+', '', filename)
        filename = re.sub(r'[()]', '', filename)[:-4]

        # Set
        labels.append(filename)
        images[i] = np_image
    
    return (images, labels, os.listdir(path))

def main(argv):
    # default values
    input_path = 'MachineLearning/NeuralNetwork/data/input'
    model_path = 'MachineLearning/NeuralNetwork/models/default.model'

    try:
        opts, args = getopt.getopt(argv,"hi:m:", ["input=", "model="])
    except:
        # ERROR
        print("Error")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # Help
            print("Usage: python .\\predict.py -i <path of your input> -m <path of your trained model>")
            sys.exit()
        elif opt in ("-i"):
            input_path = 'MachineLearning/NeuralNetwork/' + arg
        elif opt in ("-m"):
            model_path = 'MachineLearning/NeuralNetwork/' + arg

    # Load model
    print("Loading model", model_path)
    model = Convolutional()
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
    if not os.path.isdir('MachineLearning/data'):
        os.mkdir('MachineLearning/NeuralNetwork/data')

    # Prints out all inputs with the strongest prediction
    print("Labeled images will be saved to the data/results folder")
    PrintResults(results, images, filenames)

    # Prints put all the inputs in a graph visualization
    print("Graphs will be saved to the data/results folder")
    DisplayResults(results, images, labels)

if __name__ == "__main__":
   main(sys.argv[1:])