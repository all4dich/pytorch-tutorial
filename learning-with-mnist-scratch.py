from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math
import torch.nn.functional as F

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

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

n, c = x_train.shape
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

loss_func = F.cross_entropy

def model(xb):
    #return b @ weights + bias
    return log_softmax(xb @ weights + bias)


bs = 64

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

for epoch in range(epochs):
    # n : Nubmer of samples
    # bs : Batch size

    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
        print(loss_func(pred, yb), accuracy(pred, yb))