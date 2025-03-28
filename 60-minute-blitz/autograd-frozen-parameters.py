import torch
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# Finetune the model on new dataset with 10 labels
print(model.fc)
model.fc = nn.Linear(512, 10)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Notice although we register all the parameters in the optimizer, the only parameters that are computing gradients (and hence updated in gradient descent) are the weights and bias of the classifier.