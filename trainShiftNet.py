# -- coding:utf-8

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_shift import *
from dataset import BuildingDataset
import pandas as pd
from ShiftNet import ShiftNet
# from LeNet5 import LeNet5
import os
import numpy as np
from torchvision import transforms

# 固定随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    # function_test_20
    # transforms.Normalize(mean=[0.9318218, 0.9599232, 0.96266234], std=[0.13674922, 0.07559318, 0.09820192])
    # train_stn
    # transforms.Normalize(mean=[0.8988469, 0.93813044, 0.9376824], std=[0.17884357, 0.09582038, 0.13065177])
    # train_shift
    transforms.Normalize(mean=[0.89742166, 0.9391391, 0.9396217], std=[0.18003607, 0.09479778, 0.1287403])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # test_rect_train
    # transforms.Normalize(mean = [0.70381516, 0.8888911, 0.92238843], std =  [0.40284097, 0.11763619, 0.15052465])
    # valid_stn
    # transforms.Normalize(mean=[0.8980922, 0.9372042, 0.93724376], std=[0.18020877, 0.09663033, 0.13077204])
    # valid_shift
    transforms.Normalize(mean=[0.8978054, 0.9395365, 0.94048536], std=[0.17915142, 0.09420377, 0.12782192])
])

df = pd.DataFrame(columns=['loss', 'accuracy'])
# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    loss, current, n = 0.0, 0.0, 0
    test_recall = torchmetrics.Recall(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.classification.BinaryF1Score().to(device)
    model.train()
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        # print("epoch：", epoch, "的第", batch, "个inputs", X.data.size(), "labels", y.data)
        output = model(X)
        # print("epoch：", epoch, "的第", batch, "个inputs", X.data.size(), "labels", output.data)
        cur_loss = loss_fn(output, y)
        # output1 = output.squeeze(-1)
        # cur_loss = loss_fn(output1, y.float())
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, axis=1)

        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]

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
            test_F1(output.argmax(1), y)
            test_recall(output.argmax(1), y)
            test_precision(output.argmax(1), y)

            # output1 = output.squeeze(-1)
            # cur_loss = loss_fn(output1, y.float())
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
            # writer.add_scalar('Valid/valid_batch', cur_loss, epoch*len(dataloader)+n)
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

if __name__ == '__main__':
    s = f"Shift_Alexnet_fl,{train_dir},{valid_dir},batch{BATCH_SIZE},lr{LR},wd{weight_decay}"
    writer = SummaryWriter(comment=s)
    # build MyDataset
    class_sample_counts = [2464, 1053]  # train_shift
    # class_sample_counts = [7804, 603]  # train_diff
    # class_sample_counts = [3118, 381]  # train_Area or train_Rect
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 这个 get_classes_for_all_imgs是关键
    train_data = BuildingDataset(data_dir=train_dir, transform=transform)
    train_targets = train_data.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    valid_data = BuildingDataset(data_dir=valid_dir, transform=transform_val)

    # build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                              pin_memory=True, sampler=sampler)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
                              shuffle=True)
    # AlexNet model and training
    net = ShiftNet(num_classes=N_FEATURES, init_weights=True)
    # net = LeNet5(num_classes=N_FEATURES)
    # net = GoogleNet(num_class=N_FEATURES)
    # 模拟输入数据，进行网络可视化
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # with writer:
    #     writer.add_graph(net, (input_data,))pythp
    # 模型进入GPU
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     net = nn.DataParallel(net)
    net.to(device)

    # 定义损失函数（交叉熵损失）
    # 计算类别权重，权重应当是类别样本数量的倒数
    class_weights = torch.tensor([1.0 / class_sample_counts[0], 1.0 / class_sample_counts[1]], dtype=torch.float).to(device)
    # 将权重传递给损失函数
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = focal_loss(alpha= [0.29928,0.70072],gamma=2,num_classes=2)
    # loss_fn = nn.BCEWithLogitsLoss()

    # 定义优化器,SGD,
    # optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay_f)

    # 学习率按数组自定义变化
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

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
            torch.save(net.state_dict(), "save_model/shift/best_model.pth")
        # 保存最后的权重文件
        torch.save(net.state_dict(), "save_model/shift/every_model.pth")
        if t == epoch - 1:
            torch.save(net.state_dict(), "save_model/shift/last_model.pth")
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print(f'** Finished Training **')
    # df.to_csv('runs/train_shift.txt', index=True, sep=';')

