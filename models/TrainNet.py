import numpy as np
import time

import torchvision
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassRecall, MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score

from dataset import BuildingDataset
import pandas as pd
# from STNet import STNet
# from config_stn import *
from config_diff import *
# from config_shift import *
from LeNet5 import LeNet5
from AlexNet import AlexNet
from ResNet import ResNet18Binary
from VGG11 import VGG11Binary
from Vit import ViTBinary
from SimpleViT import SimpleViT
from ConvNeXt import ConvNeXt
from SwinTransformer import SwinTransformer
import os
from focal_loss import focal_loss
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from trainSTNet import convert_image_np
import os
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式

torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
torch.backends.cudnn.deterministic = False  # 牺牲可重复性换取速度

# plt.ion()
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
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    
    # rect_train
    # transforms.Normalize(mean=[0.89716554, 0.9392292, 0.9398516],std=[0.18011567, 0.09461408, 0.12847571])
    # area_train
    # transforms.Normalize(mean=[0.9333282, 0.9635094, 0.9664677], std = [0.13711655, 0.0724623, 0.09523336])
    # train_shift
    # transforms.Normalize(mean=[0.89742166, 0.9391391, 0.9396217], std=[0.18003607, 0.09479778, 0.1287403])
    # origin_diff
    transforms.Normalize(mean=[0.9250994, 0.9588776, 0.961942], std=[0.14851464, 0.07733698, 0.101837136])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    
    # rect_valid
    # transforms.Normalize(mean=[0.8994139, 0.93833137, 0.9385182], std=[0.17848423, 0.09575517, 0.12936021])
    # area_valid
    # transforms.Normalize(mean=[0.9306144, 0.9633733, 0.96703666], std = [0.13885023, 0.07191164, 0.094978005])
    # valid_shift
    # transforms.Normalize(mean=[0.8978054, 0.9395365, 0.94048536], std=[0.17915142, 0.09420377, 0.12782192])
    # valid_diff
    transforms.Normalize(mean=[0.9269143, 0.9598757, 0.96309], std=[0.14659306, 0.076451726, 0.10049462])

])

df = pd.DataFrame(columns=['loss', 'accuracy'])


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    loss, current, n = 0.0, 0.0, 0
    # 定义正则化项
    # l1_lambda = 0.0001
    # l2_lambda = 0.0001
    test_acc = MulticlassAccuracy(num_classes=N_FEATURES,average='micro').to(device)
    test_recall = MulticlassRecall(num_classes=N_FEATURES,average='none').to(device)
    test_precision = MulticlassPrecision(num_classes=N_FEATURES,average='none').to(device)
    test_F1 = MulticlassF1Score(num_classes=N_FEATURES,average='none').to(device)
    model.train()

    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(False)

    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 确保标签是long类型
        y = y.long()
        # 前向传播
        optimizer.zero_grad()
        # output, output2, output1 = model(X)
        output = model(X)

        cur_loss = loss_fn(output, y)
        cur_loss.backward()  # 反向传播
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)#梯度裁剪
        optimizer.step()  # 更新模型参数
        # cur_loss0 = loss_fn(output,y)
        # cur_loss1 = loss_fn(output1, y)
        # cur_loss2 = loss_fn(output2, y)
        # output1 = output.squeeze(-1)

        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        # 将 output 转换为预测类别（0或1）
        # pred = (output > 0).long()  # Logits > 0 预测为1，否则为0
        # pred = pred.squeeze(1)  # 从 [16, 1] 压缩为 [16]
        pred = output.argmax(dim=1)  # 获取预测的类别索引，形状: [batch_size]
        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]
        # cur_loss = cur_loss0  + cur_loss1 * 0.3 + cur_loss2 * 0.3

        test_acc.update(pred, y)
        test_F1.update(pred, y)
        test_recall.update(pred, y)
        test_precision.update(pred, y)

        # y = y.unsqueeze(dim=1).to(torch.float32)

        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
        rate = (batch + 1) / train_num
        # print(f"train loss: {rate * 100:.1f}%,{cur_loss:.3f}")
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    total_acc = test_acc.compute().item()
    print(f"train_loss' : {(loss / n):.3f}  train_acc : {(current / n):.3f}")
    print(f"accuracy of every train dataset class:{total_acc:.4f} ")
    print("recall of every train dataset class(neg/pos): ", total_recall.tolist())
    print("precision of every train dataset class(neg/pos): ", total_precision.tolist())
    print("F1 of every train dataset class(neg/pos): ", total_F1.tolist())
    writer.add_scalar('Train/Loss', loss / n, epoch)
    writer.add_scalar('Train/Acc', total_acc, epoch)
    writer.add_scalar('Train/Recall', total_recall[1], epoch)
    writer.add_scalar('Train/Precision', total_precision[1], epoch)
    writer.add_scalar('Train/F1', total_F1[1], epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    test_acc.reset()


# 定义验证函数
def val(dataloader, model, loss_fn, epoch):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    test_acc = MulticlassAccuracy(num_classes=N_FEATURES, average='micro').to(device)
    test_recall = MulticlassRecall(num_classes=N_FEATURES, average='none').to(device)
    test_precision = MulticlassPrecision(num_classes=N_FEATURES, average='none').to(device)
    test_F1 = MulticlassF1Score(num_classes=N_FEATURES, average='none').to(device)

    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 确保标签是long类型
            y = y.long()

            output = model(X) # 输出为logits（未经过sigmoid）
            # output1 = output.squeeze(-1)
            # cur_loss = loss_fn(output1, y.float())
            # 将 output 转换为预测类别（0或1）
            # pred = (output > 0).long()  # Logits > 0 预测为1，否则为0，直接以logits > 0作为分类阈值（等效于概率 > 0.5）
            # pred = pred.squeeze(1)  # 从 [16, 1] 压缩为 [16]

            # 计算准确率
            pred = output.argmax(dim=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            test_acc.update(pred, y)
            test_F1.update(pred, y)
            test_recall.update(pred, y)
            test_precision.update(pred, y)

            # y = y.unsqueeze(dim=1).to(torch.float32)
            cur_loss = loss_fn(output, y)

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    total_acc = test_acc.compute().item()
    print(f"valid_loss' : {(loss / n):.3f}  valid_acc : {(current / n):.3f}")
    print(f"accuracy of every val dataset class:{total_acc:.4f} ")
    print("recall of every val dataset class(neg/pos): ", total_recall.tolist())
    print("precision of every val dataset class(neg/pos): ", total_precision.tolist())
    print("F1 of every val dataset class(neg/pos): ", total_F1.tolist())
    writer.add_scalar('Valid/Loss', loss / n, epoch)
    writer.add_scalar('Valid/Acc', total_acc, epoch)
    writer.add_scalar('Valid/Recall', total_recall[1], epoch)
    writer.add_scalar('Valid/Precision', total_precision[1], epoch)
    writer.add_scalar('Valid/F1', total_F1[1], epoch)
    test_precision.reset()
    test_recall.reset()
    test_F1.reset()
    test_acc.reset()
    # df.loc[epoch] = {'loss': loss / n, 'accuracy': current / n}
    return current / n


# stn可视化函数
def visualize_stn(model):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(train_loader))[0].to(device)

        input_tensor = data.cpu()
        # tensor1 = model.stn(data)
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


if __name__ == '__main__':
    s = f"SwinT_diff_Pre,{train_dir},{valid_dir},batch{BATCH_SIZE},lr{LR},wd{weight_decay_f}"
    writer = SummaryWriter(comment=s)
    # build MyDataset
    # class_sample_counts = [2464, 1053]  # train_shift
    class_sample_counts = [7804, 603]  # train_diff
    # class_sample_counts = [3118, 381]  # train_Area or train_Rect

    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 这个 get_classes_for_all_imgs是关键
    train_data = BuildingDataset(data_dir=train_dir, transform=transform)
    train_targets = train_data.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    valid_data = BuildingDataset(data_dir=valid_dir, transform=transform_val)

    # build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True,
                              sampler=sampler,persistent_workers=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True
                              ,persistent_workers=True)

    # AlexNet model and training
    # net = AlexNet(num_classes=N_FEATURES, init_weights=True)
    # net = LeNet5(num_classes=N_FEATURES)
    # ---------------------New Model----------------------
    # net = ResNet18Binary()
    # net = VGG11Binary()
    # net = ViTBinary()
    # net = SimpleViT()
    # net = ConvNeXt()
    net = SwinTransformer()

    # 模拟输入数据，进行网络可视化
    # input_data = Variable(torch.rand(16, 3, 224, 224))
    # with writer:
    #     writer.add_graph(net, (input_data,))
    # 模型进入GPU
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     net = nn.DataParallel(net)

    net.to(device)

    # 定义损失函数（交叉熵损失）
    # 计算类别权重，权重应当是类别样本数量的倒数
    # class_weights = torch.tensor([1.0 / class_sample_counts[0], 1.0 / class_sample_counts[1]], dtype=torch.float).to(device)
    # class_weights = torch.tensor([1.0, 1.2], dtype=torch.float).to(device)
    # 将权重传递给损失函数
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights) #标签为整数，模型中输出N_FEATURES = 2时
    # loss_fn = focal_loss(alpha= [0.108824,0.891176],gamma=2.0,num_classes=2)
    # class_weights = torch.tensor(class_sample_counts[0] / (2*class_sample_counts[1])).to(device)#neg/pos
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)#标签为浮点数
    # loss_fn = nn.BCEWithLogitsLoss().to(device)

    # 定义优化器,SGD,
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=weight_decay_f)
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=weight_decay_f)
    # 学习率按数组自定义变化
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    # imgdata = []
    # labels = []
    # for _, (X, y) in enumerate(train_loader):
    #     imgdata.append(X)
    #     labels.append(y)
    #     # imgdata.append(torch.squeeze(X.view(1, -1)).numpy())
    #     # labels.append(torch.squeeze(y, 0).numpy().item())
    # X = torch.cat(imgdata)
    # y = torch.cat(labels)
    # # 进行网格搜索
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=4)
    # grid_search.fit(X, y)

    # 输出最佳超参数组合和对应的模型性能
    # print("Best hyperparameters of LeNet5: ", grid_search.best_params_)
    # print("Accuracy with best hyperparameters: ", grid_search.best_score_)
    # 学习率每隔10epoch变为原来的0.1
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

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
                os.mkdir('../../TrainCNN/Feature-Fusion-CNN-master/save_model')
            min_acc = a
        print('save best model', )
        torch.save(net.state_dict(), "save_model/diff_SwinT/best_model.pth")
        # torch.save(net.state_dict(), "save_model/diff/every_model.pth")
        # 保存最后的权重文件
        if t == epoch - 1:
            torch.save(net.state_dict(), "save_model/diff_SwinT/last_model.pth")
        finish = time.time()
        time_elapsed = finish - start
        print('本次训练耗时 {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    # visualize_stn(model)
    # plt.ioff()
    # plt.show()
    print(f'** Finished Training **')
    # df.to_csv('runs/train.txt', index=True, sep=';')
