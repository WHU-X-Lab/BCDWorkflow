# -- coding:utf-8
import pathlib
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class ShiftImgDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理，默认不进行预处理
        """
        # data_info存储所有图片路径和标签（元组的列表），在DataLoader中通过index读取样本
        self.get_img_info(data_dir)
        self.transform = transform
        img_paths = os.listdir(data_dir)
        img_paths = [img_path for img_path in img_paths if img_path.isdigit()]
        self.classes_for_all_imgs = []
        for img_path in img_paths:
            img_path11 = os.path.join(data_dir, img_path)
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # 添加你希望支持的图片格式
            img_paths1 = [f for f in os.listdir(img_path11) if f.endswith(valid_extensions)]
            # img_paths1 = os.listdir(img_path11)
            # print(f"Total img_paths1: {len(img_paths1)}")
            self.classes_for_all_imgs.append(img_path)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.imgs_dict):
            raise IndexError(f"Index {idx} out of range for data_info of length {len(self.imgs_dict)}")
        id_num = list(self.imgs_dict.keys())[idx]
        imgs_list = self.imgs_dict[id_num]
        for img_path, shift_label in imgs_list:
            shift_img = Image.open(img_path).convert('RGB')
            if self.transform:
                shift_img = self.transform(shift_img)
        return shift_img, id_num, shift_label

    def __len__(self):
        return len(self.imgs_dict)

        # 自定义方法，用于返回所有图片的路径以及标签

    def get_img_info(self, data_dir):
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        data_info = {}
        for sub_dir in data_dir.iterdir():
            if sub_dir.is_dir():
                for img in sub_dir.iterdir():#返回的是 pathlib.Path 对象
                    if img.suffix == '.jpg':
                        label = int(sub_dir.name) if sub_dir.name.isdigit() else -1
                        image_name = img.name  # 得到 'cat.jpg'
                        id_num = int(image_name.split('groupid')[1].split('.')[0])
                        if id_num not in data_info:
                            data_info[id_num] = []
                        data_info[id_num].append((img, label))
        self.imgs_dict = data_info
        print(f"Total data samples: {len(self.imgs_dict)}")

    def get_classes_for_all_imgs(self):
        # print(f"Total classes returned: {len(self.classes_for_all_imgs)}")
        return self.classes_for_all_imgs