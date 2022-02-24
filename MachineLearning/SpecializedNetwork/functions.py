import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys, getopt, os, time
import numpy as np
from sklearn import metrics

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
        if batch % 100 == 0:
            print(f'Loss: {loss:>7f} Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}]')

    train_loss /= num_batches
    train_acc /= size

    print(f'\n Avg Loss: {loss:>7f} Avg Acc: {(train_acc) * 100:>0.1f}%')
    
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
            if batch % 100 == 0:
                print(f'Loss: {loss:>7f} Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}]')

    val_loss /= num_batches
    val_acc /= size

    print(f'\n Avg Loss: {val_loss:>7f} Avg Acc: {(val_acc)*100:>0.1f}%')

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
    dataset = "datasets"
    
    torch.set_printoptions(profile="full")
    transform = transforms.Compose([
        transforms.RandomInvert(1),
        transforms.ToTensor()
        ]
    )
    
    # Load datasets
    train_set = Qlsa('MachineLearning/NeuralNetwork/' + dataset + '/train.csv', 'MachineLearning/NeuralNetwork/' + dataset + '/images', transform=transform)
    validation_set = Qlsa('MachineLearning/NeuralNetwork/' + dataset + '/test.csv', 'MachineLearning/NeuralNetwork/' + dataset + '/images', transform=transform)

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
    plt.xticks(np.arange(0, num_epochs+1, 1.0 if num_epochs < 50 else 10))
    plt.xlim(xmin=0)
    plt.plot(VAL_ACC, label = "Validation")
    plt.plot(TRAIN_ACC, label = "Training")
    plt.legend()

    # Loss graph
    plt.subplot(2,1,2)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, num_epochs+1, 1.0 if num_epochs < 50 else 10))
    plt.xlim(xmin=0)
    plt.plot(VAL_LOSS, label = "Validation")
    plt.plot(TRAIN_LOSS, label = "Training")
    plt.legend()

    plt.show()

def SaveModel(model, name, log):
    if not os.path.isdir('MachineLearning/NeuralNetwork/models/' + name):
        os.mkdir('MachineLearning/NeuralNetwork/models/' + name)

    path = 'MachineLearning/NeuralNetwork/models/' + name + '/' + name + '.model'
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path)

    # with open(f'MachineLearning/NeuralNetwork/models/{name}/{name}.txt', 'w+') as logfile:
    #     logfile.write(log)
