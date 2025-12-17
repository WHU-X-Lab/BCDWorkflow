# -- coding:utf-8
import torch
import numpy as np
from DifferenceNet import DifferenceNet
from ShiftNet import ShiftNet
from torchvision import transforms
import os
from PIL import Image
from GoogleNet import GoogLeNet



def extractors(img_path, save_path, net):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.891972, 0.93623203, 0.9399001], std=[0.16588122, 0.090553254, 0.12211764])
    ])
    # 打开图片，默认为PIL，需要转成RGB
    img = Image.open(img_path).convert('RGB')
    # 如果预处理的条件不为空，应该进行预处理操作
    if transform is not None:
        img = transform(img)
    image = torch.unsqueeze(img, 0).cuda()
    # 非训练，提取特征用到（模型参数不用更新， 所以no_grad）
    # print(torch.no_grad)
    with torch.no_grad():
        output = net(image)
        # print("img:", img_path, "; output", output.shape)
    output = torch.squeeze(output)
    output = output.cpu().detach().numpy()
    np.save(save_path, output)


if __name__ == '__main__':
    image_dir1 = "full_diff"
    image_dir2 = "full_shift"
    dirs = []
    label = 0
    save_path1 = 'features_diff'
    save_path2 = 'features_shift'
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    # 载入模型参数
    net1 = DifferenceNet(num_classes=2, init_weights=True)
    # net1 = GoogLeNet(num_classes=2)
    net1.load_state_dict(torch.load('save_model/different/last_model.pth'))
    net1.cuda()
    # 遍历样本数据，依次提取特征
    for jpeg in os.listdir(image_dir1):
        test_img1 = os.path.join(image_dir1, jpeg)
        save_feature_path1 = os.path.join(save_path1,jpeg)
        if os.path.exists(save_feature_path1) == False:
            os.mkdir(save_feature_path1)
        extractors(test_img1, save_feature_path1, net1)
    net2 = ShiftNet(num_classes=2, init_weights=True)
    net2.load_state_dict(torch.load('save_model/shift/last_model.pth'))
    net2.cuda()
    for jpeg in os.listdir(image_dir2):
        test_img2 = os.path.join(image_dir2, jpeg)
        save_feature_path2 = os.path.join(save_path2,jpeg)
        if os.path.exists(save_feature_path2) == False:
            os.mkdir(save_feature_path2)
        extractors(test_img2, save_feature_path2, net2)
