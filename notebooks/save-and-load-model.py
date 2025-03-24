import torch
import torchvision.models as models
import logging

log_level: str = "WARNING"
log_console_format = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s (in %(pathname)s:%(lineno)d)"
logging.basicConfig(format=log_console_format, datefmt="%m/%d/%Y %I:%M:%S %p %Z")
numeric_level = getattr(logging, log_level.upper(), None)
logging.getLogger().setLevel(numeric_level)

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model2 = models.vgg16()
model2.load_state_dict(torch.load('model_weights.pth'))
print(model2.eval())