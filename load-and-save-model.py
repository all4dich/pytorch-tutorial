import torch
import torchvision.models as models

model_download = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model_download.state_dict(), 'model_weights.pth')
torch.save(model_download, 'model_shape.pth')

model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True) )
model_2 = torch.load('model_shape.pth', weights_only=False)
model.eval()
model_2.eval()