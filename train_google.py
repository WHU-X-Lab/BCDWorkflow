import numpy as np
import time
from sklearn.compose import TransformedTargetRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from model import AlexNet
# from model import LeNet
from GoogleNet import GoogLeNet
from config_google import *
# from config_diff import *
# from config_shift import *
from dataset import BuildingDataset
import pandas as pd
import os
from focal_loss import focal_loss
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import torchmetrics
import matplotlib.pyplot as plt
from STNGooNet import STNGoogLeNet
from DifferenceNet import DifferenceNet
from LeNet5 import LeNet5
from GoogleNet import GoogLeNet
import torch.nn.functional as F

plt.ion()
# 固定随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # rect_train
    # transforms.Normalize(mean=[0.89716554, 0.9392292, 0.9398516],std=[0.18011567, 0.09461408, 0.12847571])
    # area_train
    transforms.Normalize(mean=[0.9333282, 0.9635094, 0.9664677], std = [0.13711655, 0.0724623, 0.09523336])

    # train_shift
    # transforms.Normalize(mean=[0.89742166, 0.9391391, 0.9396217], std=[0.18003607, 0.09479778, 0.1287403])
    # origin_diff
    # transforms.Normalize(mean=[0.9250994, 0.9588776, 0.961942], std=[0.14851464, 0.07733698, 0.101837136])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # rect_valid
    # transforms.Normalize(mean=[0.8994139, 0.93833137, 0.9385182], std=[0.17848423, 0.09575517, 0.12936021])
    # area_valid
    transforms.Normalize(mean=[0.9306144, 0.9633733, 0.96703666], std = [0.13885023, 0.07191164, 0.094978005])

    # valid_shift
    # transforms.Normalize(mean=[0.8978054, 0.9395365, 0.94048536], std=[0.17915142, 0.09420377, 0.12782192])
    # valid_diff
    # transforms.Normalize(mean=[0.9269143, 0.9598757, 0.96309], std=[0.14659306, 0.076451726, 0.10049462])
])

df = pd.DataFrame(columns=['loss', 'accuracy'])


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    loss, current, n = 0.0, 0.0, 0
    # test_acc = torchmetrics.Accuracy().to(device)
    test_recall = torchmetrics.Recall(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.classification.BinaryF1Score().to(device)
    model.train()
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        # print(y)
        # print(type(y))
        output, output2, output1 = model(X)
        # output = model(X)
        cur_loss0 = loss_fn(output, y)
        cur_loss1 = loss_fn(output1, y)
        cur_loss2 = loss_fn(output2, y)
        # output1 = output.squeeze(-1)
        # cur_loss = loss_fn(output1, y.float())
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]
        cur_loss = cur_loss0 + cur_loss1 * 0.3 + cur_loss2 * 0.3

        # test_acc(pred.argmax(1), y)
        # test_recall = test_recall.to(device)
        # test_precision = test_precision.to(device)
        # test_acc = test_acc.to(device)

        # test_acc(output.argmax(1), y)
        test_F1(output.argmax(1), y)
        test_recall(output.argmax(1), y)
        test_precision(output.argmax(1), y)

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
        rate = (batch + 1) / train_num
        # print(f"train loss: {rate * 100:.1f}%,{cur_loss:.3f}")
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    # total_acc = test_acc.compute()
    print(f"train_loss' : {(loss / n):.3f}  train_acc : {(current / n):.3f}")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    writer.add_scalar('Train/Loss', loss / n, epoch)
    writer.add_scalar('Train/Acc', current / n, epoch)
    writer.add_scalar('Train/Recall', total_recall.item(), epoch)
    writer.add_scalar('Train/Precision', total_precision.item(), epoch)
    writer.add_scalar('Train/F1', total_F1.item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()


# 定义验证函数
def val(dataloader, model, loss_fn, epoch):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    test_recall = torchmetrics.Recall(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.classification.BinaryF1Score().to(device)
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            # output1 = output.squeeze(-1)
            # cur_loss = loss_fn(output1, y.float())
            test_F1(output.argmax(1), y)
            test_recall(output.argmax(1), y)
            test_precision(output.argmax(1), y)

            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    print(f"valid_loss' : {(loss / n):.3f}  valid_acc : {(current / n):.3f}")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    writer.add_scalar('Valid/Loss', loss / n, epoch)
    writer.add_scalar('Valid/Acc', current / n, epoch)
    writer.add_scalar('Valid/Recall', total_recall.item(), epoch)
    writer.add_scalar('Valid/Precision', total_precision.item(), epoch)
    writer.add_scalar('Valid/F1', total_F1.item(), epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    df.loc[epoch] = {'loss': loss / n, 'accuracy': current / n}
    return current / n


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))  # 将颜色通道放到后面
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # 用公式"(x-mean)/std"，将每个元素分布到(-1,1)，也就是标准化
    inp = np.clip(inp, 0, 1)  # # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
    return inp


def attention(image):
    attention_map = torch.sum(image, dim=1).to(device)
    attention_map = attention_map / torch.max(attention_map)
    x = torch.sum(attention_map, dim=2).to(device)
    y = torch.sum(attention_map, dim=1).to(device)
    x_c = torch.sum(x * torch.arange(x.size(1)).to(device)).to(device) / (torch.sum(x, dim=1)).to(device)
    y_c = torch.sum(y * torch.arange(y.size(1)).to(device)).to(device) / (torch.sum(y, dim=1)).to(device)
    return x_c, y_c


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def visualize_stn(model):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(train_loader))[0]
        model.eval()
        # 使用STN模型得到变换后的图像和变换矩阵
        transformed_data1, theta1 = model.stn1(data.to(device))
        transformed_data2, theta2 = model.stn2(data.to(device))
        # transformed_data1, theta = model.stn(transformed_data0.to(device))
        batch_size, _, h, w = transformed_data1.shape
        # 计算图像注意力中心
        x_center, y_center = w // 2, h // 2
        theta1 = torch.cat((theta1.to(device), torch.tensor([0, 0, 1]).repeat(batch_size, 1, 1).to(device)), dim=1)
        theta2 = torch.cat((theta2.to(device), torch.tensor([0, 0, 1]).repeat(batch_size, 1, 1).to(device)), dim=1)
        # x_center2, y_center2 = attention(transformed_data2.to(device))

        # calculate the attention center coordinates for each example in the batch
        attention_center_coordinates1 = []
        attention_center_coordinates2 = []
        for i in range(batch_size):
            # 构造齐次坐标
            point = torch.tensor([x_center, y_center, 1.]).view(3, 1).to(device)
            # 计算逆变换矩阵
            inv_theta1 = torch.inverse(theta1[i]).to(device)
            inv_theta2 = torch.inverse(theta2[i]).to(device)
            # 对特征图像的中心点进行逆变换
            out_point1 = inv_theta1.matmul(point).view(-1).to(device)
            out_point2 = inv_theta2.matmul(point).view(-1).to(device)
            # 获取注意力中心在原始图像上的坐标
            x_out1 = out_point1[0].item()
            y_out1 = out_point1[1].item()
            x_out2 = out_point2[0].item()
            y_out2 = out_point2[1].item()
            attention_center_coordinates1.append([x_out1, y_out1])
            attention_center_coordinates2.append([x_out2, y_out2])

        # 创建matplotlib的Figure和Axes对象
        fig, axs = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                ncols=4,
                                figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        fig2, axs2 = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                  ncols=4,
                                  figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        fig3, axs3 = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                  ncols=4,
                                  figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        for i in range(batch_size):
            ax = axs[i // 4, i % 4]
            ax2 = axs2[i // 4, i % 4]
            ax3 = axs3[i // 4, i % 4]
            ax.imshow(convert_image_np(data[i]))
            ax2.imshow(convert_image_np(transformed_data1[i]))
            ax3.imshow(convert_image_np(transformed_data2[i]))
            attention_center_row1, attention_center_col1 = attention_center_coordinates1[i]
            attention_center_row2, attention_center_col2 = attention_center_coordinates2[i]
            attention_box_size = 64
            x0 = int(attention_center_row1 - attention_box_size / 2 - 1)
            y0 = int(attention_center_col1 - attention_box_size / 2 - 1)
            x1 = int(attention_center_row1 + attention_box_size / 2 - 1)
            y1 = int(attention_center_col1 + attention_box_size / 2 - 1)
            x2_0 = int(attention_center_row2 - attention_box_size / 2)
            y2_0 = int(attention_center_col2 - attention_box_size / 2)
            x2_1 = int(attention_center_row2 + attention_box_size / 2)
            y2_1 = int(attention_center_col2 + attention_box_size / 2)
            # x_c = x_center[i].item()
            # y_c = y_center[i].item()
            # x_c2 = x_center2[i].item()
            # y_c2 = y_center2[i].item()

            rect = plt.Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                                 fill=False,
                                 edgecolor='red',
                                 linewidth=3.5)
            rect2 = plt.Rectangle((x2_0, y2_0), x2_1 - x2_0 + 1, y2_1 - y2_0 + 1,
                                  fill=False,
                                  edgecolor='green',
                                  linewidth=3.5)
            ax.add_patch(rect)
            ax.add_patch(rect2)


import matplotlib.patches as patches


def visualize_stn2(model):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(train_loader))[0]
        model.eval()
        # 使用STN模型得到变换后的图像和变换矩阵
        transformed_data1, theta1 = model.stn1(data.to(device))
        transformed_data2, theta2 = model.stn2(data.to(device))
        # transformed_data1, theta = model.stn(transformed_data0.to(device))
        batch_size, _, h, w = transformed_data1.shape
        theta1[:, :, 0] *= 224
        theta1[:, :, 1] *= 224
        theta1 = theta1.cpu().numpy()
        theta2[:, :, 0] *= 224
        theta2[:, :, 1] *= 224
        theta2 = theta2.cpu().numpy()
        # 创建matplotlib的Figure和Axes对象
        fig, axs = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                ncols=4,
                                figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        fig2, axs2 = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                  ncols=4,
                                  figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        fig3, axs3 = plt.subplots(nrows=batch_size // 4 + (batch_size % 4 != 0),
                                  ncols=4,
                                  figsize=(15, (batch_size // 4 + (batch_size % 4 != 0)) * 3))
        for i in range(batch_size):
            ax = axs[i // 4, i % 4]
            ax2 = axs2[i // 4, i % 4]
            ax3 = axs3[i // 4, i % 4]
            ax.imshow(convert_image_np(data[i]))
            ax2.imshow(convert_image_np(transformed_data1[i]))
            ax3.imshow(convert_image_np(transformed_data2[i]))
            rect1 = patches.Rectangle(
                (theta1[i][0][1], theta1[i][0][0]),  # (y, x) 坐标
                theta1[i][1][1] - theta1[i][0][1],  # 矩形宽度
                theta1[i][1][0] - theta1[i][0][0],  # 矩形高度
                linewidth=1, edgecolor='r', facecolor='none'
            )
            rect2 = patches.Rectangle(
                (theta2[i][0][1], theta2[i][0][0]),  # (y, x) 坐标
                theta2[i][1][1] - theta2[i][0][1],  # 矩形宽度
                theta2[i][1][0] - theta2[i][0][0],  # 矩形高度
                linewidth=1, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect1)
            ax.add_patch(rect2)


if __name__ == '__main__':
    s = f"GoogleNet_Area_lf,{train_dir},{valid_dir},batch{BATCH_SIZE},lr{LR},wd{weight_decay_f}"
    writer = SummaryWriter(comment=s)
    # build MyDataset
    # class_sample_counts = [2464, 1053]  # train_shift
    # class_sample_counts = [7804, 603]  # train_diff
    class_sample_counts = [3118, 381]  # train_Area or train_Rect
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 这个 get_classes_for_all_imgs是关键
    train_data = BuildingDataset(data_dir=train_dir, transform=transform)
    train_targets = train_data.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    valid_data = BuildingDataset(data_dir=valid_dir, transform=transform_val)

    # build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                              sampler=sampler)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True)
    # AlexNet model and training
    # net = AlexNet(num_classes=N_FEATURES, init_weights=True)
    # net = DifferenceNet(num_classes=N_FEATURES, init_weights=True)
    # net = LeNet5(num_classes=N_FEATURES)
    net = GoogLeNet(num_classes=N_FEATURES,init_weights=True,aux_logits=True)
    # net = STNGoogLeNet(num_classes=N_FEATURES, init_weights=True, aux_logits=True)
    # 模拟输入数据，进行网络可视化
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # with writer:
    #     writer.add_graph(net, (input_data,))
    # 模型进入GPU 
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     net = nn.DataParallel(net)
    # net.load_state_dict(torch.load('./save_model/stn/last_model.pth'), False)
    model = net.to(device)
    # 定义损失函数（交叉熵损失）
    # 计算类别权重，权重应当是类别样本数量的倒数
    # class_weights = torch.tensor([1.0 / class_sample_counts[0], 1.0 / class_sample_counts[1]], dtype=torch.float).to(device)
    # 将权重传递给损失函数
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = focal_loss(alpha= [0.108824,0.891176],gamma=2,num_classes=2)
    # loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)

    # 定义优化器,SGD,
    # optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_f)
    optimizer = optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay_f)
    # 学习率每隔10epoch变为原来的0.1
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # 总参数计算
    # total_params = sum(p.numel() for p in net.parameters())
    # print(f"Total number of GoogleNet+2STN parameters: {total_params}")

    # 开始训练
    epoch = MAX_EPOCH
    min_acc = 0
    train_num = len(train_loader)
    for t in range(epoch):
        start = time.time()
        print(f"epoch{t + 1}\n-------------------")
        train(train_loader, net, loss_fn, optimizer, t)
        a = val(valid_loader, net, loss_fn, t)
        lr_scheduler.step()
        print("目前学习率:", optimizer.param_groups[0]['lr'])
        # 保存最好的模型权重文件
        if a > min_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            min_acc = a
            print('save best model', )
            torch.save(net.state_dict(), "save_model/area_Goo/best_model.pth")
        torch.save(net.state_dict(), "save_model/area_Goo/every_model.pth")
        # if float(a) < float(85) :
        #     torch.save(net.state_dict(), "save_model/diff_Goo/lunwen_model.pth")
        # 保存最后的权重文件
        if t == epoch - 1:
            torch.save(net.state_dict(), "save_model/area_Goo/last_model.pth")
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    # visualize_stn(model)
    # plt.ioff()
    # plt.show()
    print(f'** Finished Training **')
    # df.to_csv('runs/train_focal.txt', index=True, sep=';')
