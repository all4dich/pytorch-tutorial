# Create a list of 100 random tensors (100x100)
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST as mnist
import os
from torchvision import transforms
from torchvision.transforms import ToTensor
import cv2
# tensors = [torch.rand(100, 100, dtype=torch.float32) for _ in range(100)]
tensors = torch.randn(100, 100, 100, dtype=torch.float32)

## Create a custom dataset from the tensor list
# class TensorDataset:
#    def __init__(self, tensors):
#        self.tensors = tensors
#
#    def __len__(self):
#        return len(self.tensors)
#
#    def __getitem__(self, idx):
#        return self.tensors[idx]

# Create dataset and dataloader
dataset = TensorDataset(tensors)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

transform_norm = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)
transform = transforms.Compose(
    [transforms.ToTensor(), ]
)

mnist_dataset_origin = mnist(root=f"{os.environ['HOME']}/data/", train=True, download=True)
mnist_dataloader_origin = DataLoader(mnist_dataset_origin, batch_size=16, shuffle=True)

mnist_dataset_totensor = mnist(root=f"{os.environ['HOME']}/data/", train=True, download=True, transform=ToTensor())
mnist_dataloader_totensor = DataLoader(mnist_dataset_totensor, batch_size=16, shuffle=True)

mnist_dataset_norm = mnist(root=f"{os.environ['HOME']}/data/", train=True, download=True, transform=transform_norm)
mnist_dataloader_norm = DataLoader(mnist_dataset_norm, batch_size=16, shuffle=True)


import matplotlib.pyplot as plt

# Get the first batch from the dataloader
image_1 = mnist_dataset_totensor[0][0][0]
image_2 = image_1 ** 100

for i, (x, y) in enumerate(mnist_dataloader_totensor):
    print(i, x.shape, y.shape)
    print(y)
    print(x.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    for j in range(x.shape[0]):
        axes[j].imshow(x[j][0], cmap='gray')
        axes[j].set_title(f"Label: {y[j]}")
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    break
