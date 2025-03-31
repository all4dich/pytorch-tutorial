from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

bs = 64

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open(mode="wb").write(content)

with gzip.open(PATH / FILENAME, "rb", "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# pyplot.show()

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

n, c = x_train.shape
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


loss_func = F.cross_entropy


def model(xb):
    # return b @ weights + bias
    return log_softmax(xb @ weights + bias)



xb = x_train[0:bs]
preds = model(xb)
preds[0], preds.shape
print(preds[0], preds.shape)


# negative log likelihood
def nll(input, target):
    return - input[range(target.shape[0]), target].mean()


# loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


print(accuracy(preds, yb))

lr = 0.5
epochs = 2

from torch import nn


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        #self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        #self.bias = nn.Parameter(torch.zeros(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        #return xb @ self.weights + self.bias
        return self.lin(xb)

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16,16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16,10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)


def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    return train_dl, valid_dl

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    Lambda(lambda x: x.view(x.size(0), -1))
)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
