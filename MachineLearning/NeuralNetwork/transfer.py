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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Load Data
transform = transforms.Compose([
        transforms.RandomInvert(1),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ]
    )

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

model_conv = model.Convolutional()
model_conv.fc3 = nn.Linear(model_conv.fc3.in_features, 10)
model_conv.load_state_dict(torch.load('mnist.model'))

#criterion = nn.CrossEntropyLoss()
#optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001)
#model_conv = train_model(model_conv, criterion, optimizer_conv, scheduler=None, num_epochs=25) 
#torch.save(model_conv.state_dict(), 'mnist.model')

# Freeze the weights
for param in model_conv.parameters():
    param.requires_grad = False

# Replace the fully connected layers
model_conv.fullyconnected = nn.Sequential(
    nn.Linear(model_conv.fc1.in_features, model_conv.fc1.out_features)
)
model_conv.fc2 = nn.Sequential(
    nn.Linear(model_conv.fc2.in_features, model_conv.fc2.out_features)
)
model_conv.fc3 = nn.Sequential(
    nn.Linear(model_conv.fc3.in_features, 22)
)

# Load dataset
dataset_name = 'merged_augmented'
train = dataset.Qlsa(dataset=dataset_name, train=True, transform=transform)
validate = dataset.Qlsa(dataset=dataset_name, train=False, transform=transform)
dataloaders = {'train': DataLoader(train, batch_size, True), 'val': DataLoader(validate, batch_size, True)}
dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset)}

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.01)

# # Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=3, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25) 