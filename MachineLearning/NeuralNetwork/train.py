import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import Qlsa
from model import Convolutional

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import getopt
import time
import sys
import os

dirname = os.path.dirname(__file__)
np.set_printoptions(linewidth=200)

def main(argv):
    # Set model parameters
    learning_rate = 0.0005
    batch_size = 64
    model_name, device, num_epochs, stopping_point, dataset = GetParameters(argv)

    # Load Dataset
    validation_loader, train_loader = LoadDataset(dataset, device, batch_size)

    # Create model
    model = Convolutional(100).to(device)

    # Class Weights
    #raw = [515, 198, 113, 154, 594, 239, 194, 194, 77, 151, 365, 359, 143, 40, 242, 107, 217, 152, 337, 227, 424, 207]
    #norm = [1 - (float(i)/max(raw)) for i in raw]
    #class_weights = torch.FloatTensor(norm).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    # Check if directory exists
    if not os.path.isdir(dirname + '\\models'):
        os.mkdir(dirname + '\\models')
    if not os.path.isdir(dirname+'\\models\\'+model_name):
        os.mkdir(dirname+'\\models\\'+model_name)

    # Lists for keeping track of metrics
    train_losses, train_accuracies = list(), list()
    validation_losses, validation_accuracies = list(), list()

    # Open logfile for logging training metrics
    with open(f'MachineLearning\\NeuralNetwork\\models\\{model_name}\\log.txt', 'a+') as logfile:
        # Clear logfile, then start writing
        logfile.truncate(0)
        logfile.write(str(model))

        # Main loop
        time_start = time.time()
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            logfile.write(f'\n Epoch {epoch + 1}/{num_epochs}\n-----------------------------------------------------------------\n')
            
            # Training Loop
            model.train()
            train_loss, train_acc = TrainingLoop(train_loader, model, loss, optimizer, device, logfile)
            train_accuracies.append(train_acc* 100)
            train_losses.append(train_loss)
            
            # Validation Loop
            model.eval()
            val_loss, val_acc = ValidationLoop(validation_loader, model, loss, device, logfile)
            validation_accuracies.append(val_acc * 100)
            validation_losses.append(val_loss)

            # Early stop if using callback function
            if train_loss <= stopping_point and stopping_point != -1: 
                #TODO: add code to check if loss is approching unsignificant improvement
                num_epochs = epoch + 1
                logfile.write(f'Early stopping at {epoch + 1} epochs')
                break

        logfile.close()
    
    print(f'Elapsed time: {time.time() - time_start:>0.2f} seconds')

    # Save model
    SaveModel(model, model_name)

    # Plot errors
    PlotGraph(num_epochs, train_losses, train_accuracies, validation_losses, validation_accuracies)

def TrainingLoop(dataloader, model, loss_function, optimizer, device, logfile):
    # Log header
    logfile.write("--- Training Loop --- \n")
    
    # Training parameters
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
        
        # Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get the loss and accuracy
        loss, current = loss.item(), batch * len(image)
        train_acc += correct.item()
        train_loss += loss

        # Log Loss and Accuracy per 100 batches
        if batch % 100 == 0:
            logfile.write(f'Loss: {loss:>7f}, Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}] \n')

    # Calculate average values
    train_loss /= num_batches
    train_acc /= size

    # log average Loss and Accuracy
    logfile.write(f'Avg Loss: {loss:>7f}, Avg Acc: {(train_acc) * 100:>0.1f}% \n')
    
    return train_loss, train_acc

def ValidationLoop(dataloader, model, loss_function, device, logfile):
    # Log header
    logfile.write("\n --- Validation Loop --- \n ")

    # Training parameters
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    val_loss, val_acc, correct = 0, 0, 0
    y_true, y_pred= list(), list()
    
    # Validation loop
    with torch.no_grad():
        for batch, (image, label) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)

            # Compute prediction and loss
            pred = model(image)
            loss = loss_function(pred, label)
            correct = (pred.argmax(1) == label).type(torch.float).sum()

            # Get the true labels and predictions
            y_true.extend(label.cpu())
            y_pred.extend(pred.argmax(1).cpu())
            
            # Get loss and accuracy
            loss, current = loss.item(), batch * len(image)
            val_acc += correct.item()
            val_loss += loss

            # log loss and accuracy per 100 batches
            if batch % 100 == 0:
                logfile.write(f'Loss: {loss:>7f}, Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}] \n')

    # Calculate average loss and accuracy
    val_loss /= num_batches
    val_acc /= size

    # log average loss and accuracy
    logfile.write(f'Avg Loss: {val_loss:>7f}, Avg Acc: {(val_acc)*100:>0.1f}% \n')

    # Create a cofusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    # Log perfomance metrics
    logfile.write("\n --- Confusion Matrix --- \n")
    logfile.write(np.array2string(cm))
    logfile.write("\n\n --- Classification Report --- \n")
    logfile.write(metrics.classification_report(y_true,y_pred, zero_division=1))

    # Calculate percison and recall
    #precision = np.array([np.diag(cm) / np.sum(cm, axis=0)])
    #recall = np.array([np.diag(cm) / np.sum(cm, axis=1)])
    #precision = np.around(precision, decimals=2)
    #recall = np.around(recall, decimals=2)

    return val_loss, val_acc

def LoadDataset(dataset, device, batch_size):
    # Transform to apply to dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(100),
        transforms.Normalize(0.5, 0.5)
        ]
    )
    
    # Load datasets
    #train_set = Qlsa(dataset=dataset, train=True, transform=transform)
    #validation_set = Qlsa(dataset=dataset, train=False, transform=transform)

    train_set = torchvision.datasets.MNIST("mnist", True, transform, download=True)
    validation_set = torchvision.datasets.MNIST("mnist", False, transform, download=True)
    
    # Create data loaders
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Transfer the dataset to the correct device
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
    
    return (validation_loader, train_loader)

def PlotGraph(num_epochs, train_losses, train_accuracies, validaton_losses, validation_accuracies):
    plt.title('Validation Error per Epoch')

    # Acc graph
    plt.subplot(2,1,1)
    plt.ylabel('Accuracy')
    plt.ylim([0,110])
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, num_epochs, 1.0 if num_epochs < 50 else 10))
    plt.xlim(xmin=0)
    plt.plot(validation_accuracies, label = "Validation")
    plt.plot(train_accuracies, label = "Training")
    plt.legend()

    # Loss graph
    plt.subplot(2,1,2)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, num_epochs, 1.0 if num_epochs < 50 else 10))
    plt.xlim(xmin=0)
    plt.plot(validaton_losses, label = "Validation")
    plt.plot(train_losses, label = "Training")
    plt.legend()

    plt.show()

def SaveModel(model, name):
    # Save model to the specified directory
    path = dirname+ '\\models\\' + name + '\\' + name + '.model'
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path)

def GetParameters(argv):
    num_epochs = 20
    stopping_point = -1
    model_name = 'default'
    device = torch.device("cpu")
    dataset = "default"

    try:
        opts, args = getopt.getopt(argv,"hm:e:d:", ["dataset=", "model=", "confusion", "report", "gpu", "earlystop="])
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
            "--gpu: if you'd like to train with GPU\n")
            
            sys.exit()
        elif opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("-e"):
            try:
                num_epochs = int(arg)
            except:
                print("argument needs to be a number")
                sys.exit(2)
        elif opt in ("--earlystop"):
            stopping_point = int(arg)
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("--gpu"):
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('GPU not available, using CPU')
                device = torch.device("cpu")
    
    return (model_name, device, num_epochs, stopping_point, dataset)

if __name__ == "__main__":
   main(sys.argv[1:])