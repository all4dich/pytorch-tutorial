import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = torch.device('cpu')
print(f"Using {device} device")

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for image classification.

    This network consists of the following layers:
    - Flatten: Flattens the input image into a 1D tensor.
    - Linear + ReLU: Fully connected layer with ReLU activation.
    - Linear + ReLU: Fully connected layer with ReLU activation.
    - Linear: Fully connected layer that outputs class scores.

    Methods
    -------
    forward(x):
        Defines the forward pass of the network.
    """

    def __init__(self):
        """
        Initialize the neural network layers.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        Perform the forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing a batch of images.

        Returns
        -------
        torch.Tensor
            Output tensor containing the class scores for each image.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Train the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader for the training data.
    model : nn.Module
        The neural network model to be trained.
    loss_fn : callable
        Loss function to be used.
    optimizer : torch.optim.Optimizer
        Optimizer to update the model parameters.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()  # Adjust the parameters by the gradients collected in the backward pass
        optimizer.zero_grad()  # Reset the gradients after updating the parameters

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loopo(dataloader, model, loss_fn):
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader for the test data.
    model : nn.Module
        The neural network model to be evaluated.
    loss_fn : callable
        Loss function to be used.
    """
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test errro: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 30
start = time.time()
for t in range(epochs):
    print(f"Epoch {t + 1}\n------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loopo(test_dataloader, model, loss_fn)

end = time.time()
print(f"Done! {end - start:.2f} seconds elapsed.")
