from torch import nn
from torchvision.models import vit_b_16

class ViTBinary(nn.Module):
    def __init__(self):
        super(ViTBinary, self).__init__()
        self.model = vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 1)
    def forward(self, x):
        return self.model(x)