# -- coding:utf-8
# Feature Fusion
import os
import random
import string
from collections import defaultdict
import time

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, compute_sample_weight
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
from ResNet import ResNet18Binary
from ConvNeXt import ConvNeXt
from SwinTransformer import SwinTransformer
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import torch
from torchvision import transforms
from PIL import Image
# from ShiftNet import ShiftNet
# from DifferenceNet import DifferenceNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from imblearn.over_sampling import ADASYN
from ShiftImgDataset import ShiftImgDataset
from DiffImgDataset import DiffImgDataset
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 固定随机数种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

N_FEATURES = 2
device = 'cuda:0'
difffeatures_dir = 'features_diff' # （1）随机复制
# difffeatures_dir = 'features_0diff'  # （2）缺失维度补0
shiftfeatures_dir = 'features_shift'
shiftimg_dir = 'full_shift'
diffimg_dir = 'full_diff'


# difffeatures_dir = 'features_SwinT/test_diff' # （1）随机复制
# shiftfeatures_dir = 'features_SwinT/test_shift'
# difffeatures_dir = 'features_Conv/test_diff' # （1）随机复制
# shiftfeatures_dir = 'features_Conv/test_shift'
# shiftimg_dir = 'raw_pic/shift'
# diffimg_dir = 'raw_pic/diff'

output_dir = 'gradcam'
# 定义图像转换
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # train_shift
    # transforms.Normalize(mean=[0.89731014, 0.9391688, 0.9396487], std=[0.18013711, 0.09479259, 0.12887895])
    # origin_diff
    # transforms.Normalize(mean=[0.9251727, 0.95890087, 0.9619809], std=[0.14847293, 0.07731944, 0.101800375])
])
transform_val  = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # valid_shift
    # transforms.Normalize(mean=[0.8978054, 0.9395365, 0.94048536], std=[0.17915142, 0.09420377, 0.12782192])
])

class DiffImgDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.imgs_dict = {}
        for img in self.imgs:
            # if img == '.DS_Store':
            #     continue
            id_n = int(img.split('groupid')[1].split('diff')[0])
            label_diff = int(img.split('diff')[1].split('x')[0])
            if id_n not in self.imgs_dict:
                self.imgs_dict[id_n] = []
            self.imgs_dict[id_n].append((img, label_diff))

    def __len__(self):
        return len(self.imgs_dict)

    def __getitem__(self, idx):
        id_n = list(self.imgs_dict.keys())[idx]
        imgs_list = self.imgs_dict[id_n]
        if len(imgs_list) > 4:
            imgs_list = sorted(imgs_list, key=lambda x: x[1], reverse=True)[:4]
        elif len(imgs_list) < 4:
            while len(imgs_list) < 4:
                # imgs_list.append(("null", 0))  # 方法一：补0
                # 方法二：随机复制
                imgs_list.append(random.choice(imgs_list))
        imgs = []  # 存储图像的张量
        for img_name, label_d in imgs_list:
            # --------------补0-----------------
            # if img_name != "null":
            #     img_path = os.path.join(self.img_dir, img_name)
            #     img = Image.open(img_path).convert('RGB')
            #     if self.transform:
            #         img = self.transform(img)
            # elif img_name == "null":
            #     img = torch.zeros([3, 224, 224])
            # -------------随机复制----------------
            img_path = os.path.join(self.img_dir,img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            imgs.append(img)
        return torch.stack(imgs), id_n




class ShiftImgDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.data_path = img_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.imgs_dict = {}
        for img in self.imgs:
            id_num = int(img.split('groupid')[1].split('ex')[0])
            label_num = int(img.split('ex')[1].split('.')[0])
            if id_num not in self.imgs_dict:
                self.imgs_dict[id_num] = []
            self.imgs_dict[id_num].append((img, label_num))

    def __len__(self):
        return len(self.imgs_dict)

    def __getitem__(self, index):
        # print(f"Dataset size: {len(self.imgs_dict)}")
        # 获取所有 key 列表
        keys = list(self.imgs_dict.keys())
        # 确保索引在有效范围内
        if index >= len(keys):
            raise IndexError(f"Index {index} out of range for dataset size {len(keys)}")
        id = keys[index]
        imgs_list = self.imgs_dict[id]
        for img_name, label in imgs_list:
            img_path = os.path.join(self.data_path, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, id, label
# def apply_gradcam(model, target_layer, input_tensor, rgb_img, target_class=None):
#     """
#     应用Grad-CAM生成热力图
#     Args:
#         model: 目标模型
#         target_layer: 要分析的层（如model.layer4）
#         input_tensor: 输入张量 (1, C, H, W)
#         rgb_img: 原始图像 (H, W, 3) [0-1范围]
#     Returns:
#         heatmap: Grad-CAM热力图
#     """
#     model.eval()
#     cam = GradCAM(model=model, target_layers=[target_layer])
#     # 自动确定目标类别
#     if target_class is None:
#         with torch.no_grad():
#             output = model(input_tensor)
#             target_class = output.argmax(dim=1).item()
#     input_tensor.requires_grad = True
#     grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])
#     # 优化后处理
#     grayscale_cam = postprocess_cam(grayscale_cam)
#     heatmap = show_cam_on_image(rgb_img, grayscale_cam[0,:], use_rgb=True)
#     return heatmap
#
#
# def postprocess_cam(cam):
#     """后处理CAM图以减少边缘噪声"""
#     # 归一化
#     cam = np.maximum(cam, 0)
#     cam = cam / (cam.max() + 1e-8)
#     # 高斯平滑
#     cam = cv2.GaussianBlur(cam, (11, 11), 5)
#     # 阈值处理(保留前20%的激活)
#     # threshold = np.percentile(cam, 80)
#     # cam[cam < threshold] = 0
#     # 重新归一化
#     cam = cam / (cam.max() + 1e-8)
#     return cam
#
# def extract_gradcam(diff_model, dataloader,method_name):
#     """
#        提取特征并保存Grad-CAM可视化结果
#        Args:
#            method_name: "zero_padding" 或 "random_replicate"
#        """
#     diff_model.eval()
#     for param in diff_model.parameters():
#         param.requires_grad = True
#     target_layer = diff_model.model.layer4[-1].conv2  # ResNet18，最后一层卷积
#     for data in dataloader:
#         imgs, id_nums = data
#         batch_size = imgs.size(0)
#         imgs = imgs.view(-1, 3, 224, 224)
#         # 原始图像预处理
#         rgb_imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
#         rgb_imgs = (rgb_imgs - rgb_imgs.min()) / (rgb_imgs.max() - rgb_imgs.min()+1e-5)
#         # 前向传播
#         with torch.set_grad_enabled(True):  # 必须启用梯度
#             output = diff_model(imgs)
#             pred_classes = output.argmax(dim=1)
#         for i in range(batch_size):
#             # 生成并保存热力图
#             heatmap = apply_gradcam(
#                 model=diff_model,
#                 target_layer=target_layer,
#                 input_tensor=imgs[i].unsqueeze(0),
#                 rgb_img=rgb_imgs[i],
#                 target_class=pred_classes[i].item()
#             )
#             cv2.imwrite(f"{output_dir}/cam_id{id_nums[i].item()}_{method_name}.jpg", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))


def extract_features(diff_model, dataloader):
    features_dict = {}
    diff_model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, id_nums = data
            batch_size = imgs.size(0)
            imgs = imgs.view(-1, 3, 224, 224)
            features = diff_model(imgs)
            features = features.view(batch_size, -1)
            for i in range(batch_size):
                features_dict[id_nums[i].item()] = (features[i].cpu().numpy())
    return features_dict

def extract_features2(shift_model, dataloader):
    features_dict = {}
    shift_model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, id_nums, labels = data
            batch_size = imgs.size(0)
            imgs = imgs.view(-1, 3, 224, 224)
            features = shift_model(imgs)
            features = features.view(batch_size, -1)
            for i in range(batch_size):
                features_dict[id_nums[i].item()] = (features[i].cpu().numpy(), labels[i].item())
    return features_dict

def features_fusion(shiftfeature_path,difffeature_path):
    # 遍历文件夹中的所有文件
    fusion_features_dict = {}
    # 准备标签值
    labels_dict = {}
    for file_name in tqdm(os.listdir(shiftfeature_path)):
        # 检查文件是否为 .npy 文件
        if file_name.endswith('.npy'):
            # 提取 id
            id = file_name.split('id')[1].split('ex')[0]
            label = int(file_name.split('ex')[1].split('.')[0])
            # 读取特征
            diff_filename = f'id{id}.npy'
            shift_feature = np.load(os.path.join(shiftfeature_path, file_name))
            if diff_filename in os.listdir(difffeature_path):
                diff_feature = np.load(os.path.join(difffeature_path, diff_filename))
                # 融合特征
                fusion_features_dict[id] = np.concatenate((shift_feature, diff_feature),
                                                           axis=0)  # shape: (num_samples, 1024+4608)
                labels_dict[id] = label  # shape: (num_samples,)
    return fusion_features_dict, labels_dict

            # if id in features_shift:
            #     fusion_features_dict[id] = (
            #     np.concatenate((features_shift[id][0], difffeature), axis=0), features_shift[id][1])
            # else:
            #     fusion_features_dict[id] = (difffeature, features_shift[id][1])



diffdataset = DiffImgDataset(img_dir=diffimg_dir, transform=transform_train)
diffimgloader = DataLoader(dataset=diffdataset, batch_size=16)
shiftdataset = ShiftImgDataset(img_dir=shiftimg_dir,transform=transform_train)
shiftimgloader = DataLoader(dataset=shiftdataset, batch_size=16, drop_last=True)

# 加载预训练模型
# 载入模型参数
# DiffNet = ResNet18Binary()
# ShiftNet = ResNet18Binary()
DiffNet = SwinTransformer() #Swin Transformer特征维度: 768
ShiftNet = SwinTransformer()
# DiffNet = ConvNeXt()
# ShiftNet = ConvNeXt()
# 加载保存的模型权重（忽略全连接层参数不匹配）
DiffNet.load_state_dict(torch.load('save_model/diff_SwinT/last_model.pth'), strict=False)
ShiftNet.load_state_dict(torch.load('save_model/shift_SwinT/best_model.pth'), strict=False)
# DiffNet.load_state_dict(torch.load('save_model/diff_ConvNeXt/best_model.pth'), strict=False)
# ShiftNet.load_state_dict(torch.load('save_model/shift_ConvNeXt/last_model.pth'), strict=False)

# extract_gradcam(diff_model=DiffNet, dataloader=diffimgloader,method_name='random_sample')
# extract_gradcam(diff_model=DiffNet, dataloader=diffimgloader,method_name='padding_zero')

# features_diff = extract_features(diff_model=DiffNet, dataloader=diffimgloader)
# for id_num in tqdm(features_diff):
#     feature_path = os.path.join(difffeatures_dir, f'id{id_num}.npy')
#     np.save(feature_path, features_diff[id_num])
# #
# features_shift = extract_features2(shift_model=ShiftNet, dataloader=shiftimgloader)
# for id_num in tqdm(features_shift):
#     feature_path = os.path.join(shiftfeatures_dir, f'id{id_num}ex{features_shift[id_num][1]}.npy')
#     np.save(feature_path, features_shift[id_num][0])


# 准备 SVM 输入数据
fusion_features_train, labels_train = features_fusion(shiftfeature_path=shiftfeatures_dir, difffeature_path=difffeatures_dir)

X_train = []
X_val = []
y_train = []
y_val = []

# 合并所有特征和标签
all_features = {}
all_labels = {}
# 合并训练集特征
for idt in fusion_features_train:
    all_features[idt] = fusion_features_train[idt]
    all_labels[idt] = labels_train[idt]

# 重新划分数据集（80%训练，20%验证）
from sklearn.model_selection import train_test_split
import numpy as np
# 获取所有ID并打乱
all_ids = list(all_features.keys())
np.random.shuffle(all_ids)
# 提取特征和标签
X_all = [all_features[id] for id in all_ids]
y_all = [all_labels[id] for id in all_ids]
# 按8:2比例划分
X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
    X_all, y_all, all_ids, test_size=0.2, random_state=42, stratify=y_all
)

# for idt in tqdm(fusion_features_train):
#     X_train.append(fusion_features_train[idt])
#     y_train.append(labels_train[idt])  # 根据您的数据设置标签
# for idv in tqdm(fusion_features_val):
#     X_val.append(fusion_features_val[idv])
#     y_val.append(labels_val[idv])  # 根据您的数据设置标签

# print('GoogLeNet+Diff/AlexNet+Shift: (1)随机复制✔(2)补0')
# print('ResNet: (1)随机复制(2)补0✔')
# ----------------------SVM--------------------------
# # 分割数据
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # 计算类别权重
# class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# cw = dict(enumerate(class_weight))
# # writer = SummaryWriter()
# # 训练 SVM 模型并输出结果
# clf = svm.SVC(class_weight=cw)
# clf.fit(X_train, y_train)
# y_train_pred = clf.predict(X_train)
# y_test_pred = clf.predict(X_test)
#
# # 计算分类评价指标
# target_names = ['class 0', 'class 1']
# print(classification_report(y_train, y_train_pred, target_names=target_names,digits=4))
# print(classification_report(y_test, y_test_pred, target_names=target_names,digits=4))

# ----------------------NN--------------------------
X_train_tensor = torch.from_numpy(np.array(X_train))
y_train_tensor = torch.from_numpy(np.array(y_train))
X_val_tensor = torch.from_numpy(np.array(X_val))
y_val_tensor = torch.from_numpy(np.array(y_val))
# 获取训练集和验证集的特征和标签
class_sample_counts = [2698, 342]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
# samples_weights = weights[y_train]
samples_weights = compute_sample_weight('balanced', y_train)
# sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(y_train)*2, replacement=True)
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
assert len(train_dataset) > 0, "训练集不能为空"
assert len(val_dataset) > 0, "验证集不能为空"
# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=True)
# 检查数据集大小

# # 统计正类占比
# train_pos_ratio = sum(y_train) / len(y_train)
# val_pos_ratio = sum(y_val) / len(y_val)
# print(f"训练集正类占比: {train_pos_ratio:.2f}，验证集正类占比: {val_pos_ratio:.2f}")
#
# # 检查数据是否重复
# for i, (inputs, labels) in enumerate(train_loader):
#     print(f"Batch {i} 输入均值: {inputs.mean().item()}")
#     if i > 2: break


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
        # self.fc1 = nn.Linear(26880,672)#Conv
        self.dropout = nn.Dropout(p=0.8)
        # self.fc2 = nn.Linear(672, 2)
        self.fc2 = nn.Linear(768, 2)

    def forward(self, x):
        # x = self.layers(x)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x


model = NNet().to(device)

# 定义损失函数和优化器
# pos_weight = torch.tensor([class_sample_counts[0] / class_sample_counts[1]]).to(device)
# class_weights = torch.tensor([1.0, 8.0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss().to(device)
# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,70,90], gamma=0.1)
s = f"SwinT_Classify,{difffeatures_dir},{shiftfeatures_dir},batch{batch_size}"
writer = SummaryWriter(comment=s)
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    loss, current, n = 0, 0, 0
    start = time.time()
    print(f"epoch{epoch + 1}\n-------------------")
    # 训练阶段
    # test_recall = torchmetrics.Recall(task="binary",average='none', num_classes=N_FEATURES).to(device)
    # test_precision = torchmetrics.Precision(task="binary",average='none', num_classes=N_FEATURES).to(device)
    # test_F1 = torchmetrics.classification.BinaryF1Score().to(device)
    test_acc = MulticlassAccuracy(num_classes=N_FEATURES, average='micro').to(device)
    test_recall = MulticlassRecall(num_classes=N_FEATURES, average='none').to(device)
    test_precision = MulticlassPrecision(num_classes=N_FEATURES, average='none').to(device)
    test_F1 = MulticlassF1Score(num_classes=N_FEATURES, average='none').to(device)
    model.train()
    for i, (features, labels) in enumerate(train_loader):
        # 前向传播
        features, labels = features.to(device), labels.long().to(device)
        outputs = model(features)
        # 将 output 转换为预测类别（0或1）
        # pred = (outputs > 0).long()  # Logits > 0 预测为1，否则为0
        # pred = pred.squeeze(1)  # 从 [64, 1] 压缩为 [64]
        cur_loss = criterion(outputs, labels)
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(outputs.data, 1)

        # test_F1(outputs.argmax(1), labels)
        # test_recall(outputs.argmax(1), labels)
        # test_precision(outputs.argmax(1), labels)
        test_acc.update(pred, labels)
        test_F1.update(pred, labels)
        test_recall.update(pred, labels)
        test_precision.update(pred, labels)

        # labels = labels.unsqueeze(dim=1).to(torch.float32)
        # cur_loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        cur_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += (pred == labels).sum().item()
        n += labels.size(0)
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
    # 验证阶段
    model.eval()
    valloss, valcorrect,valtotal, n = 0.0, 0.0, 0.0, 0
    val_acc = MulticlassAccuracy(num_classes=N_FEATURES, average='micro').to(device)
    val_recall = MulticlassRecall(num_classes=N_FEATURES, average='none').to(device)
    val_precision = MulticlassPrecision(num_classes=N_FEATURES, average='none').to(device)
    val_F1 = MulticlassF1Score(num_classes=N_FEATURES, average='none').to(device)
    with torch.no_grad():
        for valfeatures, vallabels in val_loader:
            valfeatures, vallabels = valfeatures.to(device), vallabels.long().to(device)
            valoutputs = model(valfeatures)
            # valpred = (valoutputs > 0).long()  # Logits > 0 预测为1，否则为0，直接以logits > 0作为分类阈值（等效于概率 > 0.5）
            # valpred = valpred.squeeze(1)
            valpred = torch.max(valoutputs, 1)[1]
            val_acc.update(valpred, vallabels)
            val_F1.update(valpred, vallabels)
            val_recall.update(valpred, vallabels)
            val_precision.update(valpred, vallabels)
            # vallabels = vallabels.unsqueeze(dim=1).to(torch.float32)
            val_loss = criterion(valoutputs, vallabels)
            _, predicted = torch.max(valoutputs.data, 1)
            valtotal += vallabels.size(0)
            valloss += val_loss.item()
            valcorrect += (predicted == vallabels).sum().item()
    print(f"val_loss: {(valloss / valtotal):.3f}  val_acc : {(valcorrect / valtotal):.3f}")
    valtotal_recall = val_recall.compute()
    valtotal_precision = val_precision.compute()
    valtotal_F1 = val_F1.compute()
    valtotal_acc = val_acc.compute().item()
    print(f"accuracy of every Valid dataset class:{valtotal_acc:.4f} ")
    print("recall of every Valid dataset class(neg/pos): ", valtotal_recall.tolist())
    print("precision of every Valid dataset class(neg/pos): ", valtotal_precision.tolist())
    print("F1 of every Valid dataset class(neg/pos): ", valtotal_F1.tolist())
    writer.add_scalar('Valid/Loss', valloss / valtotal, epoch)
    writer.add_scalar('Valid/Acc', valtotal_acc, epoch)
    writer.add_scalar('Valid/Recall', valtotal_recall[1], epoch)
    writer.add_scalar('Valid/Precision', valtotal_precision[1], epoch)
    writer.add_scalar('Valid/F1', valtotal_F1[1], epoch)
    val_precision.reset()
    val_recall.reset()
    val_F1.reset()
    val_acc.reset()


    # lr_scheduler.step()
    # 保存最好的模型权重文件
    min_acc = 0
    vala = valcorrect/valtotal
    if vala > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = vala
        print('save best model', )
        torch.save(model.state_dict(), "save_model/classify/Conv/best_model.pth")
    # 保存最后的权重文件
    torch.save(model.state_dict(), "save_model/classify/Conv/every_model.pth")
    # if epoch == 149:
        # torch.save(model.state_dict(), "save_model/classify/Conv/last_model.pth")
    finish = time.time()
    time_elapsed = finish - start
    print('本次训练耗时 {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

print(f'** Finished Training **')