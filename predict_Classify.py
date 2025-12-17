# -- coding:utf-8
# Feature Fusion
import os
import random
import string
from collections import defaultdict

import torchmetrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import torch
from torchvision import transforms
from PIL import Image
from ShiftNet import ShiftNet
from DifferenceNet import DifferenceNet
from GoogleNet import GoogLeNet
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 固定随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

N_FEATURES = 2
device = 'cuda:0'
# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # train_shift
    # transforms.Normalize(mean=[0.89731014, 0.9391688, 0.9396487], std=[0.18013711, 0.09479259, 0.12887895])
    # origin_diff
    # transforms.Normalize(mean=[0.9251727, 0.95890087, 0.9619809], std=[0.14847293, 0.07731944, 0.101800375])
])


class DiffImgDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('diff')[0])
            label_num = int(img.split('diff')[1].split('x')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs_dict)

    def __getitem__(self, idx):
        id_num = list(self.imgs_dict.keys())[idx]
        imgs_list = self.imgs_dict[id_num]
        if len(imgs_list) > 4:
            imgs_list = sorted(imgs_list, key=lambda x: x[1], reverse=True)[:4]
        elif len(imgs_list) < 4:
            while len(imgs_list) < 4:
                # imgs_list.append(("null", 0))  # 方法一：补0
                imgs_list.append(random.choice(imgs_list)) # 方法二：随机复制
        imgs = []
        for img_name, label in imgs_list:
            # 补0
            # if img_name is not "null":
            #     img_path = os.path.join(self.img_dir, img_name)
            #     img = Image.open(img_path).convert('RGB')
            #     if self.transform:
            #         img = self.transform(img)
            # elif img_name is "null":
            #     img = torch.zeros([3, 224, 224])
            # 随机复制
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            imgs.append(img)
        return torch.stack(imgs), id_num

shiftimg_dir = 'full_shift'
class ShiftImgDataset(Dataset):
    def __init__(self, data_path=shiftimg_dir, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.imgs = os.listdir(data_path)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('ex')[0])
            label_num = int(img.split('ex')[1].split('.')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        id = list(self.imgs_dict.keys())[index]
        imgs_list = self.imgs_dict[id]
        for img_name, label in imgs_list:
            img_path = os.path.join(self.data_path, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, id, label


def extract_features(model, dataloader):
    features_dict = {}
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, id_nums = data
            batch_size = imgs.size(0)
            imgs = imgs.view(-1, 3, 224, 224)
            features = model(imgs)
            features = features.view(batch_size, -1)
            for i in range(batch_size):
                features_dict[id_nums[i].item()] = (features[i].cpu().numpy())  # , labels[i].item())
    return features_dict


# 定义模型
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(5120, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1))
        self.fc1 = nn.Linear(3840, 768)#SwinT
        # self.fc1 = nn.Linear(26880,672)
        self.dropout = nn.Dropout(p=0.7)
        # self.fc2 = nn.Linear(672, 2)
        self.fc2 = nn.Linear(768, 2)

    def forward(self, x):
        # x = self.layers(x)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # image_dir = "pick_predict/area"
    # difffeatures_dir = 'features_SwinT/test_diff' # （1）随机复制
    # shiftfeatures_dir = 'features_SwinT/test_shift'
    difffeatures_dir = 'features_Conv/test_diff' # （1）随机复制
    shiftfeatures_dir = 'features_Conv/test_shift'

    # 遍历文件夹中的所有文件
    fusion_features_dict = {}
    # 准备标签值
    labels_dict = {}
    for file_name in tqdm(os.listdir(shiftfeatures_dir)):
        # 检查文件是否为 .npy 文件
        if file_name.endswith('.npy'):
            # 提取 id
            id = file_name.split('id')[1].split('ex')[0]
            label = int(file_name.split('ex')[1].split('.')[0])
            if label == 0:
                label = 0
            elif label != 0:
                label = 1
            # 读取特征
            diff_filename = f'id{id}.npy'
            shift_feature = np.load(os.path.join(shiftfeatures_dir, file_name))
            if diff_filename in os.listdir(difffeatures_dir):
                diff_feature = np.load(os.path.join(difffeatures_dir, diff_filename))
                # 融合特征
                fusion_features_dict[id] = np.concatenate((shift_feature, diff_feature),
                                                        axis=0)  # shape: (num_samples, 1024+4608)
                labels_dict[id] = label  # shape: (num_samples,)

            # if id in features_shift:
            #     fusion_features_dict[id] = (
            #     np.concatenate((features_shift[id][0], difffeature), axis=0), features_shift[id][1])
            # else:
            #     fusion_features_dict[id] = (difffeature, features_shift[id][1])

    # 准备 SVM 输入数据
    X = []
    y = []
    ids = []
    for id in tqdm(fusion_features_dict):
        # print("X", fusion_features_dict[id])
        # print("y", labels_dict[id])
        ids.append(id)
        X.append(fusion_features_dict[id])
        y.append(labels_dict[id])  # 根据您的数据设置标签
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # 输入特征
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)     # 真实标签
    model = NNet()
    # model.load_state_dict(torch.load('save_model/classify/SwinT/best_model.pth'))
    model.load_state_dict(torch.load('save_model/classify/Conv/best_model.pth'))
    model.to(device)
    model.eval()

    # test_recall = torchmetrics.Recall(task="binary",average='none', num_classes=N_FEATURES).to(device)
    # test_precision = torchmetrics.Precision(task="binary",average='none', num_classes=N_FEATURES).to(device)
    # test_F1 = torchmetrics.classification.BinaryF1Score().to(device)
    # test_confusion = torchmetrics.ConfusionMatrix(task="binary",num_classes=N_FEATURES).to(device)
    with torch.no_grad():
        # 对 X_tensor 进行预测
        outputs = model(X_tensor)
        # 二分类问题，使用 sigmoid 函数将输出转换为概率
        predictions = torch.sigmoid(outputs)
        # 将概率大于等于 0.5 的预测为类别 1，小于 0.5 的预测为类别 0
        predicted_classes = (predictions >= 0.5).long()
        # test_F1(predicted_classes, y_tensor)
        # test_recall(predicted_classes, y_tensor)
        # test_precision(predicted_classes, y_tensor)
        # test_confusion(predicted_classes, y_tensor)
    
    for i in range(len(ids)):
        print(f"图像'{ids[i]}'预测为1的概率为 {predictions[i][1].item():.4f}，真实类别为 {y[i]}")

    # total_recall = test_recall.compute()
    # total_precision = test_precision.compute()
    # total_F1 = test_F1.compute()
    # total_confusion = test_confusion.compute()
    # print("recall of every test dataset class: ", total_recall)
    # print("precision of every test dataset class: ", total_precision)
    # print("F1 of every test dataset class: ", total_F1)
    # print("confusion matrix: ", total_confusion)
    # print(f"预测正确率为:{(i / n) * 100:.3f}%")
