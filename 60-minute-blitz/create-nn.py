import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operaiton: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels, 5x5 square convolution kernel
        # It uses ReLU activation function, and
        # outptus a Tensor with size ( N, 6, 28, 28)
        c1 = F.relu(self.conv1(input))
        # Subsmapling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameters, and outputs a (N, 6, 14, 14) tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input image channel, 6 output channels, 5x5 square convolution kernel
        # It uses ReLU activation function, and
        # outptus a Tensor with size ( N, 16, 10, 10)
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output

net = Net()
image_size = 32
input = torch.randn(1, 1, image_size, image_size)
out = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print("-------------------")
net.zero_grad()
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)

# Run optimization manually
learning_rate = 1e-2
##for f in net.parameters():
##    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer.zero_grad()
loss = criterion(out, target)
loss.backward()
optimizer.step()