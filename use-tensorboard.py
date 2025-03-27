from torch.utils.tensorboard import SummaryWriter
import torch
import math

num_epochs = 10
writer = SummaryWriter("runs/test_logger_5")

for epoch in range(num_epochs):
    train_loss = torch.randn(1)[0]
    val_accuracy = torch.randn(1)[0]

#    writer.add_scalar('Loss/train', train_loss, epoch)
#    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
#    writer.add_scalar('A', epoch *2 , epoch)
#    writer.add_scalar('B', epoch * 3, epoch)

x = torch.linspace(0., 2 * math.pi, 100)
for i in range(x.size()[0]):
    X = x[i]
    writer.add_scalar('sin', math.sin(X), i)
    writer.add_scalar('cos', math.cos(X), i)
    writer.add_scalar('tan', math.tan(X), i)
    writer.add_scalars("all", {"sin": math.sin(X), "cos": math.cos(X), "tan": math.tan(X)}, i)
writer.close()
