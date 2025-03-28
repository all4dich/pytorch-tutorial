import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()
print("Backward pass done")
# Learning rate 0.01 , Momentum 0.9
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Initiate gradient descent
optim.step()
print("Optimization done")