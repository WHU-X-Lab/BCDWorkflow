# -- coding:utf-8
import random

import torch
from torch.utils.data import DataLoader
import cv2
import os
from config_diff import *
from torchvision import transforms
from dataset import BuildingDataset
from LeNet5 import LeNet5
from GoogleNet import GoogLeNet
from AlexNet import AlexNet
from DifferenceNet import DifferenceNet
from ShiftNet import ShiftNet
import torch.nn as nn
import torchmetrics.classification

# import numpy as np

# import matplotlib.pyplot as plt
# npydir = r'E:\Naraku\Feature-Fusion-CNN\features\diff0topid0osmid54547x146761.jpg.npy'
# depthmap = np.load(npydir)    #使用numpy载入npy文件
# plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# # plt.colorbar()                   #添加colorbar
# plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# plt.show()                        #在线显示图像

# -----------------------------Test----------------------------
# if torch.backends.mps.is_available():
#     device = torch.device(device)
#     print("GPU is available")
# else:
#     device = torch.device("cpu")

# ---------------以下为202409生成示例图用的predict代码-------------
def img_preprocess(raw_image):
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            # compareValid
            # transforms.Normalize(mean=[0.84808147, 0.9096524, 0.91053045], std=[0.20253241, 0.10373465, 0.14877456])

        ])(raw_image[..., ::-1].copy())
    image = torch.unsqueeze(image, 0).to(device)  # 1*3*224*224
    return image

# 定义输出标签的函数
def val(model, test_data_img, label):
    # 将模型转为验证模式
    img = cv2.imread(test_data_img)
    img_input = img_preprocess(img)
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    with torch.no_grad():
        output = model(img_input)
        # cur_loss = loss_fn(output, y)
        output = torch.nn.functional.softmax(output/2.0, dim=1)
        # print(output)
        _, pred = torch.max(output, axis=1)
    # pred = nn.Softmax(pred)
    # print(f"第一次{pred}")
    pred = pred.cpu()
    # print(f"'第二次'+'{pred}'")
    pred = pred.numpy()
    # print(f"'第三次'+'{pred}'")
    p = pred[0]
    # if int(p) != int(label):
        # cv2.imwrite(save_wrong, img)
    # else:
        # cv2.imwrite(save_right, img)
    return [p, output]


if __name__ == '__main__':
    # image_dir = "pick_predict/area"
    image_dir = "valid_area"
    dirs = []
    label = 0
    # save_wrong = 'save_predict/shift_wrong'
    # save_right = 'save_predict/shift_right'

    #LeNet
    # net1 = LeNet5(num_classes=N_FEATURES)
    # state_dict = torch.load("save_model/area_Le/best_model.pth")
    # net1.load_state_dict(state_dict)

    # googlenet模型验证
    net1 = GoogLeNet(num_classes=2,init_weights=True,aux_logits=True)
    # # if torch.cuda.device_count() > 1:
    # #     print("Use", torch.cuda.device_count(), 'gpus')
    # #     net1 = nn.DataParallel(net1)
    net1.load_state_dict(torch.load("save_model/area_Goo/last_model.pth"))

    # alexnet模型验证
    # net1 = AlexNet(num_classes=2, init_weights=True)
    # state_dict = torch.load("save_model/area_Alex/last_model.pth")
    # net1.load_state_dict(state_dict)

    net1.to(device)

    net1.eval()
    for dir in os.listdir(image_dir):  # 遍历数据中的标签目录文件名 {0，1}
        if dir == '.DS_Store':
            continue
        dirs.append(dir)
    dirs.sort()
    i = 0
    n = 0
    test_recall = torchmetrics.Recall(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(task="binary",average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.classification.BinaryF1Score().to(device)
    test_confusion = torchmetrics.ConfusionMatrix(task="binary",num_classes=N_FEATURES).to(device)
    for dir in dirs:
        # if dir == '0' :
        #     continue
        cur_dir = os.path.join(image_dir, dir)
        files = []
        for entry in os.listdir(cur_dir):
            files.append(entry)
        # n = n + int(len(files))
        # 计算需要选择的文件数量
        num_files_to_select = max(1, int(len(files) * 1))  # 至少选择一个文件
        # 随机选择 1% 的文件
        selected_files = random.sample(files, num_files_to_select)
        n = n + int(len(selected_files))
        for jpeg in selected_files:
            test_img = os.path.join(cur_dir, jpeg)
            # save_wrong_img = os.path.join(save_wrong, jpeg)
            # save_right_img = os.path.join(save_right, jpeg)
            pred,output = val(net1, test_img, dir)  # 返回的是【pred, output】
            # print(type(dir))
            temp = int(dir)
            temp1 = [temp]
            temp2 = list(temp1)
            y = torch.tensor(temp2).to(device)
            y = y.to(device)
            # print(y)
            test_F1(output.argmax(1), y)
            test_recall(output.argmax(1), y)
            test_precision(output.argmax(1), y)
            test_confusion(output.argmax(1), y)
            prob_0 = round(output[0, 0].item(),4)  # 提取第一个类别的概率
            prob_1 = round(output[0, 1].item(),4)  # 提取第二个类别的概率
            print(f"图像'{test_img}'预测为0的概率为 {prob_0}(预测为{pred}，真实类别为 {dir})")
            if str(pred) == dir:
                i = i + 1
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    total_confusion = test_confusion.compute()
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    print("confusion matrix: ", total_confusion)
    print(f"预测正确率为:{(i / n) * 100:.3f}%")
