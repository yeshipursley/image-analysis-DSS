import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import HebrewDataset
from model import HebrewNet

# # Load dataset
# mnist_train = datasets.MNIST(
#     root="./datasets", 
#     train=True, 
#     transform=transforms.ToTensor(), 
#     download=True
# )

# mnist_test = datasets.MNIST(
#     root="./datasets", 
#     train=False, 
#     transform=transforms.ToTensor(), 
#     download=True
# )

# # DataLoaders
# train_loader = torch.utils.data.DataLoader(
#     mnist_train, 
#     batch_size=64, 
#     shuffle = True
# )

# test_loader = torch.utils.data.DataLoader(
#     mnist_test, 
#     batch_size=64, 
#     shuffle = True
# )

# Hyperparameters
learning_rate = 0.001
batch_size = 10

num_epochs = 10

training_set = HebrewDataset('datasets/dataset.csv', 'datasets/images', transform=transforms.ToTensor())
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

# Create model
model = HebrewNet()
model.train()

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

def train_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    for batch, (image, label) in enumerate(dataloader):
        # Compute prediction and loss
        prediction = model(image)
        print(label)
        loss = loss_function(prediction, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(image)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for image, label in dataloader:
            pred = model(image)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}\n---------------------')
    train_loop(train_loader, model, loss, optimizer)
    # test_loop(test_loader, model, loss)

torch.save(model.state_dict(), 'number.model')
print('Model saved')

#https://pytorch.org/tutorials/beginner/basics/intro.html