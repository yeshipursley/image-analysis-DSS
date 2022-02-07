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

from dataset import HebrewDataset
from model import Convolutional, Linear

TRAIN_LOSS = list()
TRAIN_ACC = list()
VAL_ACC = list()
VAL_LOSS = list()

def TrainingLoop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    training_loss, correct = 0, 0

    for batch, (image, label) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(image)
        loss = loss_function(pred, label)
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(image)
            training_loss += loss
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

    TRAIN_LOSS.append(training_loss/num_batches)
    TRAIN_ACC.append((correct/size) * 100)

def ValidationLoop(dataloader, model, loss_fn, p_c, p_r):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0
    y_true, y_pred = list(), list()
    
    with torch.no_grad():
        for image, label in dataloader:
            pred = model(image)
            validation_loss += loss_fn(pred, label).item()
            y_true.extend(label)
            y_pred.extend(pred.argmax(1))
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    validation_loss /= num_batches
    correct /= size

    print("--- Validation Error ---\n" +
    f"Accuracy  : {(100*correct):>0.1f}% \n" +
    f"Val loss  : {validation_loss:>8f} \n")

    if p_c:
        print("\n --- Confusion Matrix ---")
        print(metrics.confusion_matrix(y_true, y_pred))

    if p_r:
        print("\n --- Classification Report ---")
        print(metrics.classification_report(y_true, y_pred, zero_division=True))

    VAL_ACC.append(100*correct)
    VAL_LOSS.append(validation_loss)

def LoadDataset(batch_size):
    # Load datasets
    train_set = HebrewDataset('datasets/train.csv', 'datasets/train', transform=transforms.ToTensor())
    validation_set = HebrewDataset('datasets/test.csv', 'datasets/test', transform=transforms.ToTensor())

    # Create data loaders
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return (validation_loader, train_loader)

def PlotGraph():
    plt.title('Validation Error per Epoch')

    # Acc graph
    plt.subplot(1,2,1)
    plt.ylabel('Accuracy')
    plt.ylim([0,100])
    plt.xlabel('Epochs')
    plt.xticks(range(num_epochs))
    plt.plot(range(num_epochs), VAL_ACC, label = "Validation")
    plt.plot(range(num_epochs), TRAIN_ACC, label = "Training")
    plt.legend()

    # Loss graph
    plt.subplot(1,2,2)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xticks(range(num_epochs))
    plt.plot(range(num_epochs), VAL_LOSS, label = "Validation")
    plt.plot(range(num_epochs), TRAIN_LOSS, label = "Training")
    plt.legend()

    plt.show()

def main(argv):
    # Hyperparameters
    learning_rate = 0.01
    batch_size = 64
    num_epochs = 20

    model_name = 'default'
    print_confusion_matrix = False
    print_classification_report = False

    try:
        opts, args = getopt.getopt(argv,"hm:e:", ["model=", "confusion", "report"])
    except:
        # ERROR
        print("Error")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # Help 
            print("Usage: ")
            sys.exit()
        elif opt in ("-m"):
            model_name = arg
        elif opt in ("-e"):
            try:
                num_epochs = int(arg)
            except:
                print("argument needs to be a number")
                exit(2)
        elif opt in ("--confusion"):
            print_confusion_matrix = True
        elif opt in ("--report"):
            print_classification_report = True

    validation_loader, train_loader = LoadDataset(batch_size)

    # Create model
    model = Convolutional()
    model.train()

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    time_start = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}\n---------------------')
        TrainingLoop(train_loader, model, loss, optimizer)
        ValidationLoop(validation_loader, model, loss, print_confusion_matrix, print_classification_report)

    print("Elapsed time: " + str(time.time() - time_start))

    # Check if directory exists
    if not os.path.isdir('models'):
            os.mkdir('models')

    # Save model
    path = 'models/' + model_name + '.model'
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path)

if __name__ == "__main__":
   main(sys.argv[1:])