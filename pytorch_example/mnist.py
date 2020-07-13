from pathlib import Path
import requests

data_path = Path("data")
path = data_path / "mnist"

path.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (path / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (path / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((path / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
        return log_softmax(xb @ weights + bias)

bs = 64
xb = x_train[0:bs]
preds = model(xb)
preds[0], preds.shape

def nll(input, target):
        return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))

def acc(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

print(acc(preds, yb))

from IPython.core.debugger import set_trace

lr = 0.5
epochs = 2

for epochs in range(epochs):
        for i in range((n-1) // bs+1):
                strat_i = i * bs
                end_i =  strat_i + bs
                xb = x_train[strat_i:end_i]
                yb = y_train[strat_i:end_i]
                pred = model(xb)
                loss = loss_func(pred, yb)

                loss.backward()
                with torch.no_grad():
                        weights -= weights.grad * lr
                        bias -= bias.grad * lr
                        weights.grad.zero_()
                        bias.grad.zero_()

print(loss_func(model(xb), yb), acc(model(xb), yb))


from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

with torch.no_grad():
    for p in model.parameters():
            p -= p.grad * lr
    model.zero_grad()

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

for xb,yb in train_dl:
    pred = model(xb)

################################################################
#CNN
################################################################
import torch.nn.functional as F
import torch.optim

class mnist_cnn(nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
                self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

        def forward(self, xb):
                xb = xb.view(-1, 1, 28, 28)
                xb = F.relu(self.conv1(xb))
                xb = F.relu(self.conv2(xb))
                xb = F.relu(self.conv3(xb))
                xb = F.avg_pool2d(xb, 4)
                return xb.view(-1, xb.size(1))

lr = 0.1

from torch import optim

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

model = mnist_cnn()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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

fit(epochs, model, loss_func, opt, train_dl, valid_dl)