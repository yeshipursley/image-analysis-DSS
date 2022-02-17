from turtle import left
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys, getopt
import time
import os
import numpy as np
from sklearn import metrics
import json

from dataset import Qlsa
from model import Convolutional

TRAIN_LOSS = list()
TRAIN_ACC = list()
VAL_ACC = list()
VAL_LOSS = list()

def TrainingLoop(dataloader, model, loss_function, optimizer, device):
    print("--- Training Loop ---")
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    train_loss, train_acc, correct = 0, 0, 0

    # Main loop
    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)
        # Compute prediction and loss
        pred = model(image)
        loss = loss_function(pred, label)
        correct = (pred.argmax(1) == label).sum()
        

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(image)
        train_acc += correct.item()
        train_loss += loss
        if batch % 10 == 0:
            print(f'Loss: {loss:>7f} Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}]')

    train_loss /= num_batches
    train_acc /= size

    print(f'Avg Loss: {loss:>7f} Avg Acc: {(train_acc) * 100:>0.1f}%')
    
    TRAIN_ACC.append(train_acc* 100)
    TRAIN_LOSS.append(train_loss)
    
    return train_loss

def ValidationLoop(dataloader, model, loss_function, p_c, p_r, device):
    print("--- Validation Loop ---")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    val_loss, val_acc, correct = 0, 0, 0
    y_true, y_pred = list(), list()
    
    with torch.no_grad():
        for batch, (image, label) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)
            pred = model(image)
            loss = loss_function(pred, label)
            correct = (pred.argmax(1) == label).type(torch.float).sum()

            y_true.extend(label.cpu())
            y_pred.extend(pred.argmax(1).cpu())

            loss, current = loss.item(), batch * len(image)
            val_acc += correct.item()
            val_loss += loss
            if batch % 10 == 0:
                print(f'Loss: {loss:>7f} Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}]')

    val_loss /= num_batches
    val_acc /= size

    print(f'Avg Loss: {val_loss:>7f} Avg Acc: {(val_acc)*100:>0.1f}%')

    if p_c:
        print("\n --- Confusion Matrix ---")
        print(metrics.confusion_matrix(y_true, y_pred))

    if p_r:
        print("\n --- Classification Report ---")
        print(metrics.classification_report(y_true, y_pred, zero_division=True))

    VAL_ACC.append(val_acc*100)
    VAL_LOSS.append(val_loss)

    return val_loss

def LoadDataset(device, batch_size):
    # Load datasets
    train_set = Qlsa('MachineLearning/NeuralNetwork/datasets/train.csv', 'MachineLearning/NeuralNetwork/datasets/images', transform=transforms.ToTensor())
    validation_set = Qlsa('MachineLearning/NeuralNetwork/datasets/test.csv', 'MachineLearning/NeuralNetwork/datasets/images', transform=transforms.ToTensor())

    # Create data loaders
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
    
    return (validation_loader, train_loader)

def PlotGraph(num_epochs):
    plt.title('Validation Error per Epoch')

    # Acc graph
    plt.subplot(2,1,1)
    plt.ylabel('Accuracy')
    plt.ylim([0,110])
    plt.xlabel('Epochs')
    plt.xticks(range(num_epochs))
    plt.plot(VAL_ACC, label = "Validation")
    plt.plot(TRAIN_ACC, label = "Training")
    plt.legend()

    # Loss graph
    plt.subplot(2,1,2)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xticks(range(num_epochs))
    plt.plot(VAL_LOSS, label = "Validation")
    plt.plot(TRAIN_LOSS, label = "Training")
    plt.legend()

    plt.show()

def SaveModel(model, name, epochs):
    if not os.path.isdir('MachineLearning/NeuralNetwork/models/' + name):
        os.mkdir('MachineLearning/NeuralNetwork/models/' + name)

    path = 'MachineLearning/NeuralNetwork/models/' + name + '/' + name + '.model'
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path)

    # stats = {
    #     "epochs": epochs,
    #     "train acc": TRAIN_ACC[epochs - 1],
    #     "train loss": TRAIN_LOSS[epochs - 1],
    #     "val acc ": VAL_ACC[epochs - 1],
    #     "val loss": VAL_LOSS[epochs - 1]
    # }

    # with open('MachineLearning/NeuralNetwork/models/models.json', 'r') as infile:
    #     json_object = json.load(infile)

    # found = False
    # for model in json_object['models']:
    #     if(model['name'] == name):
    #         model['stats'] = stats
    #         found = True
    #         break
        
    # if not found:
    #     json_object['models'].append(
    #         {
    #             'name': name, 
    #             'stats': stats
    #         })
        
    # with open('MachineLearning/NeuralNetwork/models/models.json', 'w') as outfile:
    #     json.dump(json_object, outfile, indent=2)

def main(argv):
    # Hyperparameters
    learning_rate = 0.0005
    batch_size = 64
    num_epochs = 20
    stopping_point = -1

    model_name = 'default'
    print_confusion_matrix = False
    print_classification_report = False
    device = torch.device("cpu")

    try:
        opts, args = getopt.getopt(argv,"hm:e:", ["model=", "confusion", "report", "gpu", "earlystop="])
    except:
        # ERROR
        print("Error")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # Help 
            print("Usage example: train.py -m my_model -e 20 --gpu")
            print("Options: \n", 
            "-m <name>: name the final model\n",
            "-e <epochs>: amount of epochs to run\n", 
            "--gpu: if you'd like to train with GPU\n",
            "--confusion: for printing the confusion matrix every epoch\n",
            "--report: for printing the classification results every epoch\n")
            sys.exit()
        elif opt in ("-m"):
            model_name = arg
        elif opt in ("-e"):
            try:
                num_epochs = int(arg)
            except:
                print("argument needs to be a number")
                sys.exit(2)
        elif opt in ("--confusion"):
            print_confusion_matrix = True
        elif opt in ("--report"):
            print_classification_report = True
        elif opt in ("--earlystop"):
            stopping_point = int(arg)
        elif opt in ("--gpu"):
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('GPU not available, using CPU')
                device = torch.device("cpu")

    validation_loader, train_loader = LoadDataset(device, batch_size)

    # Create model
    model = Convolutional().to(device)
    model.train()

    # Weights
    #weights = [228, 104, 19, 74, 251, 106, 9, 50, 16, 137, 64, 104, 186, 66, 12, 50, 33, 28, 28, 78, 208, 48]
    #noramlWeights = [1 - (x/sum(weights)) for x in weights]
    #noramlWeights = torch.FloatTensor(noramlWeights).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    time_start = time.time()
    for epoch in range(num_epochs):
        print(f'\n Epoch {epoch + 1}/{num_epochs}\n---------------------')
        running_train_loss = TrainingLoop(train_loader, model, loss, optimizer, device)
        running_val_loss = ValidationLoop(validation_loader, model, loss, print_confusion_matrix, print_classification_report, device)

        if running_train_loss <= stopping_point and stopping_point != -1:
            num_epochs = epoch + 1
            print(f'Early stopping at {epoch + 1} epochs')
            break


    print(f'Elapsed time: {time.time() - time_start:>0.2f} seconds')

    # Check if directory exists
    if not os.path.isdir('MachineLearning/NeuralNetwork/models'):
            os.mkdir('MachineLearning/NeuralNetwork/models')

    # Save model
    SaveModel(model, model_name, num_epochs)

    # Plot errors
    PlotGraph(num_epochs)

if __name__ == "__main__":
   main(sys.argv[1:])