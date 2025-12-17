from torch import nn
import torch.nn.functional as F
import torch

# 定义一个网络模型类
class STNLeNet(nn.Module):
    # 初始化网络
    def __init__(self, num_classes=2):
        super(STNLeNet, self).__init__()
        # N=(W-F+2P)/S+1
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
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
        # return x1, theta
        return x1, x2,theta,theta2
    def forward(self, x):
        x1, x2,_,_ = self.stn(x)
        # x1, theta = self.stn(x)
        # x, _ = self.stn(x)
        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.conv1(x))  # (3,32,32) -> (16,28,28)
        x = self.pool1(x)  # (16,28,28) -> (16,14,14)
        x = F.relu(self.conv2(x))  # (16,14,14) -> (32,10,10)
        x = self.pool2(x)  # (32,10,10) -> (32,5,5)
        x = x.view(-1, 32 * 53 * 53)  # (32,5,5) -> 35*5*5
        x = F.relu((self.fc1(x)))  # 120
        x = F.relu((self.fc2(x)))  # 84
        x = self.fc3(x)  # 10
        return x
