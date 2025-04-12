import math
import torch

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias  = torch.zeros(10, requires_grad=True)
