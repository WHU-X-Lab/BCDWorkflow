# -- coding:utf-8
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


# 原本基于1x28x28图片设计的网络，需改stn结构为3*224*224
class STNet(nn.Module):
    def __init__(self, num_classes=2, init_weights=False):
        super(STNet, self).__init__()
        self.theta_fixed = 0
        self.features = nn.Sequential(
            nn.Conv2d(6, 48, kernel_size=11, stride=4, padding=2),  # (224+4-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6],
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        if init_weights:
            self._initialize_weights()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),  # 按计算结果修改
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward functioån
    def stn(self, x):
        # Localisation net
        xs = self.localization(x)  # 卷积
        xs = xs.view(-1, 10 * 52 * 52)  # resize Tensor维度重构
        theta = self.fc_loc(xs)  # 全连接（6个参数）
        theta = theta.view(-1, 2, 3)
        # print("修改前的theta：", theta)
        # theta[:, 0:1, 1:2] = self.theta_fixed  # 只位移和缩放，不旋转
        # theta[:, 1:, 0:1] = self.theta_fixed
        # print("修改后的theta:", theta)
        # Grid generator
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        # Sampler
        # x = F.grid_sample(x, grid)
        x1 = F.grid_sample(x, grid, align_corners=True)
        # 两个并行的STN
        # Localisation net
        xs2 = self.localization(x)  # 卷积
        xs2 = xs2.view(-1, 10 * 52 * 52)  # resize Tensor维度重构
        theta2 = self.fc_loc(xs2)  # 全连接（6个参数）
        theta2 = theta2.view(-1, 2, 3)
        # print("修改前的theta：", theta)
        # theta2[:, 0:1, 1:2] = self.theta_fixed  # 只位移和缩放，不旋转
        # theta2[:, 1:, 0:1] = self.theta_fixed
        # print("修改后的theta:", theta)
        # Grid generator
        grid2 = F.affine_grid(theta2, x.size(), align_corners=True)
        # Sampler
        x2 = F.grid_sample(x, grid2, align_corners=True)
        # return x, theta
        return x1, x2

    def forward(self, x):
        # transform the input
        x1,x2= self.stn(x)
        # x1, x2, theta = self.stn(x)
        x = torch.cat((x1, x2), dim=1)
        # x = self.stn(x)
        # Perform the usual forward pass（transformer后再丢到模型里训练）
        x = self.features(x)  # 特征提取
        x = torch.flatten(x, start_dim=1)  # 拉伸为一维
        x = self.classifier(x)  # 全连接
        return x

        # 参数初始化

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
