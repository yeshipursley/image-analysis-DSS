import model, dataset

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import time, copy
# Hyperparameters
batch_size = 64
device = 'cpu'

def TrainingLoop(dataloader, model, loss_function, optimizer, device):
    # Log header
    print("--- Training Loop --- \n")
    
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
            print(f'Loss: {loss:>7f}, Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}] \n')

    # Calculate average values
    train_loss /= num_batches
    train_acc /= size

    # log average Loss and Accuracy
    print(f'Avg Loss: {loss:>7f}, Avg Acc: {(train_acc) * 100:>0.1f}% \n')
    
    return train_loss, train_acc

def ValidationLoop(dataloader, model, loss_function, device):
    # Log header
    print("\n --- Validation Loop --- \n ")

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
                print(f'Loss: {loss:>7f}, Acc: {(correct/batch_size)*100:>0.1f}% [{current:>5d}/{size:>5d}] \n')

    # Calculate average loss and accuracy
    val_loss /= num_batches
    val_acc /= size

    # log average loss and accuracy
    print(f'Avg Loss: {val_loss:>7f}, Avg Acc: {(val_acc)*100:>0.1f}% \n')

    # Calculate percison and recall
    #precision = np.array([np.diag(cm) / np.sum(cm, axis=0)])
    #recall = np.array([np.diag(cm) / np.sum(cm, axis=1)])
    #precision = np.around(precision, decimals=2)
    #recall = np.around(recall, decimals=2)

    return val_loss, val_acc


# Load Model
#model_conv = model.Convolutional()
#for param in model_conv.parameters():
#    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#num_ftrs = model_conv.fc.in_features
#model_conv.fc = nn.Linear(num_ftrs, 128)
#model_conv = model_conv.to(device)

## Train initial model on MNIST
#train = torchvision.datasets.MNIST(root='mnist', train=True, download=True,transform=transform)
#validate = torchvision.datasets.MNIST(root='mnist', train=False, download=True,transform=transform)

#dataloaders = {'train': DataLoader(train, batch_size, True), 'val': DataLoader(validate, batch_size, True)}
#dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset)}

model_conv = torchvision.models.resnet18(pretrained=True)
print(model_conv)
#model_conv.load_state_dict(torch.load(r'MachineLearning\NeuralNetwork\models\mnist\mnist.model'))
#criterion = nn.CrossEntropyLoss()
#optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001)
#model_conv = train_model(model_conv, criterion, optimizer_conv, scheduler=None, num_epochs=25) 
#torch.save(model_conv.state_dict(), 'mnist.model')

# Freeze the weights
for param in model_conv.parameters():
    param.requires_grad = False

# Replace the last fully connected layers
model_conv.fc = nn.Linear(model_conv.fc.in_features, 22)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ]
    )

# Load dataset
dataset_name = 'default'
train = dataset.Qlsa(dataset=dataset_name, train=True, transform=transform)
validate = dataset.Qlsa(dataset=dataset_name, train=False, transform=transform)
dataloaders = {'train': DataLoader(train, batch_size, True), 'val': DataLoader(validate, batch_size, True)}
dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset)}

# Optimizer & Loss
optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

# Main loop
time_start = time.time()
for epoch in range(40):
    print(f'Epoch {epoch + 1}/{40}')

    # Training Loop
    model_conv.train()
    train_loss, train_acc = TrainingLoop(dataloaders['train'], model_conv, loss, optimizer, device)

    # Validation Loop
    model_conv.eval()
    val_loss, val_acc = ValidationLoop(dataloaders['val'], model_conv, loss, device)

print(f'Elapsed time: {time.time() - time_start:>0.2f} seconds')

