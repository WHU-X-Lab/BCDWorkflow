from torch import nn
from torchvision import models


class VGG11Binary(nn.Module):
    def __init__(self):
        super(VGG11Binary, self).__init__()
        self.model = models.vgg11_bn(weights=None)
        self.model.classifier[6] = nn.Linear(4096, 1)
    def forward(self, x):
        return self.model(x)