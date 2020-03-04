import torch
import torchvision.models as models
from thop import profile
from torchstat import stat

model = models.resnet50()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
flops = flops / 1e9
params = params / 1e6

model = models.resnet50()
stat(model, (3, 224, 224))