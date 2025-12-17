# ResNet18
import torch
import torch.nn as nn
from torchvision import models

class ResNet18Binary(nn.Module):
    def __init__(self):
        super(ResNet18Binary, self).__init__()
        self.model = models.resnet18(weights=None)
        # 修改全连接层为二分类
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        # 修改模型输出为特征
        # self.model.fc = torch.nn.Identity()  # 替换全连接层为恒等映射
        # 添加自适应特征层
        # self.adaptor = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5))

    # def forward(self, x):
        # x = self.model(x)
        # return self.adaptor(x)  # 输出维度1024
    def forward(self, x):
        return self.model(x)