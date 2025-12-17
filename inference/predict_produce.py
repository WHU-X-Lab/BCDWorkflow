#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from platform import freedesktop_os_release
from PIL import Image
from pickle import FALSE
from cv2 import mean
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import os
from config_diff import *
from torchvision import transforms
from dataset import BuildingDataset
from DifferenceNet import DifferenceNet
import torch.nn as nn
import torchmetrics
# 用于生成预测错误的数据，将预测的类别分别存放于相应标签的文件夹中

def img_preprocess(raw_image):
    raw_image = cv2.resize(raw_image, (224,) * 2)
    # image = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])(raw_image[..., ::-1].copy())
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.70381516, 0.8888911, 0.92238843], std =  [0.40284097, 0.11763619, 0.15052465]) #  test_rect_test
            # transforms.Normalize(mean = [0.72859555, 0.8993313, 0.9314646], std =  [0.39795846, 0.11729997, 0.14610958]) # test_rect_val
            # transforms.Normalize(mean = [0.85530734, 0.9493153, 0.96933824], std = [0.28664276, 0.087741055, 0.098699085]) # test_area_val
            # transforms.Normalize(mean=[0.84431416, 0.9468392, 0.96895444], std = [0.2916492, 0.08882934, 0.098634705]) # test_area_val
             # valid_test2
            # transforms.Normalize(mean = [0.8904361, 0.9363585, 0.94014686], std = [0.165799, 0.09002642, 0.12338792])
            #compareTrain
            # transforms.Normalize(mean = [0.84432864, 0.9061879, 0.90614516], std = [0.20210263, 0.10464828, 0.15093093])
            # compareValid
            transforms.Normalize(mean = [0.84808147, 0.9096524, 0.91053045], std = [0.20253241, 0.10373465, 0.14877456])
            # function_test_5_val
            # transforms.Normalize(mean=[0.67125416, 0.8375651, 0.8712284], std = [0.14200182, 0.064011395, 0.1304606])
            #function_test_10
            # transforms.Normalize(mean=[0.8215626, 0.8954368, 0.9014965], std = [0.19241227, 0.100563586, 0.14778145])
            # function_test_20
            # transforms.Normalize(mean= [0.92881185, 0.95924205, 0.96284324], std=[0.13795583, 0.07532157, 0.09843103])
        ])(raw_image[..., ::-1].copy())
    image = torch.unsqueeze(image, 0).cuda()				# 3
    return image


# 定义输出标签的函数
def val(model, test_data_img,label,save_path):
    # 将模型转为验证模式
    img = cv2.imread(test_data_img)
    img_input = img_preprocess(img)
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
            output = model(img_input)
            # cur_loss = loss_fn(output, y)
            # print(output)
            _, pred = torch.max(output, axis=1)
    # pred = nn.Softmax(pred)
    # print(f"第一次{pred}")
    pred = pred.cpu()
    # print(f"'第二次'+'{pred}'")
    pred = pred.numpy()
    # print(f"'第三次'+'{pred}'")
    p = pred[0]
    if int(p) != int(label):
        cv2.imwrite(save_path, img)
    return [p,output]

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    image_dir = "compareValid"
    dirs = []
    label = 0
    save_path = 'predict_wrong'


    # # googlenet模型验证
    # net1 = GoogLeNet(num_classes=2,init_weights=True,aux_logits=True)
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     net1 = nn.DataParallel(net1)
    # net1.to(device)
    # # net1.load_state_dict(torch.load('D:\变化探测研究\训练好的模型\\202112222100\last_model.pth'))
    # net1.load_state_dict(torch.load(r'use_model\google_rect_best\best_model.pth'))



    # alexnet模型验证 
    net1 = DifferenceNet(num_classes=2,init_weights=True)
    net1.load_state_dict(torch.load(r'use_model\alex_rect_last\last_model.pth'))
    net1.cuda()
    
    
    net1.eval()	
    for dir in os.listdir(image_dir): # 遍历数据中的标签目录文件名 {0，1}
        dirs.append(dir)
    dirs.sort()
    i = 0
    n = 0
    test_recall = torchmetrics.Recall(average='none', num_classes=N_FEATURES).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=N_FEATURES).to(device)
    test_F1 = torchmetrics.F1(num_classes=N_FEATURES,average='none').to(device)
    test_confusion = torchmetrics.ConfusionMatrix(num_classes= N_FEATURES).to(device)
    for dir in dirs:
        # if dir == '0' :
        #     continue
        cur_dir = os.path.join(image_dir, dir) 
        n = n + len(os.listdir(cur_dir))
        for jpeg in os.listdir(cur_dir):
            test_img = os.path.join(cur_dir, jpeg)
            save_path_img = os.path.join(save_path,dir,jpeg)
            pred = val(net1, test_img,dir,save_path_img) #返回的是【pred, output】
            # print(type(dir))
            temp = int(dir)
            temp1 = [temp]
            temp2 = list(temp1)
            y = torch.tensor(temp2)
            y = y.to(device)
            # print(y)
            output = pred[1]
            test_F1(output.argmax(1), y)
            test_recall(output.argmax(1), y)
            test_precision(output.argmax(1), y)
            test_confusion(output.argmax(1), y)
            print(f"图像'{test_img}'预测为 {pred[0]}(真实类别为 {dir})")
            if str(pred[0])==dir:
                i = i + 1 
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_F1 = test_F1.compute()
    total_confusion = test_confusion.compute()
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1 of every test dataset class: ", total_F1)
    print("confusion matrix: ", total_confusion)
    print(f"预测正确率为:{(i/n)*100:.3f}%")