# -- coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F


# 定义一个网络模型类
class LeNet5(nn.Module):
    # 初始化网络
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        # N=(W-F+2P)/S+1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (3,32,32) -> (16,28,28)
        x = self.pool1(x)  # (16,28,28) -> (16,14,14)
        x = F.relu(self.conv2(x))  # (16,14,14) -> (32,10,10)
        x = self.pool2(x)  # (32,10,10) -> (32,5,5)
        x = x.view(-1, 32 * 53 * 53)  # (32,5,5) -> 35*5*5
        x = F.relu((self.fc1(x)))  # 120
        x = self.dropout(x)  # 在训练时应用 Dropout
        x = F.relu((self.fc2(x)))  # 84
        x = self.fc3(x)  # 10
        return x
