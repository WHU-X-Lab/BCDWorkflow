# -- coding:utf-8
import pathlib
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class DiffImgDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理，默认不进行预处理
        """
        # data_info存储所有图片路径和标签（元组的列表），在DataLoader中通过index读取样本
        self.get_img_info(data_dir)
        self._balance_data()
        self.transform = transform
        img_paths = os.listdir(data_dir)
        img_paths = [img_path for img_path in img_paths if img_path.isdigit()]
        self.classes_for_all_imgs = []
        for img_path in img_paths:
            img_path11 = os.path.join(data_dir, img_path)
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # 添加你希望支持的图片格式
            img_paths1 = [f for f in os.listdir(img_path11) if f.endswith(valid_extensions)]
            self.classes_for_all_imgs.append(img_path)

    def _balance_data(self):
        """预处理阶段平衡数据"""
        balanced_dict = {}
        for id_num, imgs_list in self.imgs_dict.items():
            current_imgs = imgs_list.copy()
            if len(current_imgs) > 4:
                balanced_dict[id_num] = random.sample(current_imgs, 4)
            elif len(current_imgs) < 4:
                while len(current_imgs) < 4:
                    # 方法一：补0
                    current_imgs.append(("null", 0))
                    # 方法二：随机复制
                    # current_imgs.append(random.choice(current_imgs))
                balanced_dict[id_num] = current_imgs
            else:
                balanced_dict[id_num] = current_imgs
        self.imgs_dict = balanced_dict

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.imgs_dict):
            raise IndexError(f"Index {idx} out of range for data_info of length {len(self.imgs_dict)}")
        id_num = list(self.imgs_dict.keys())[idx]
        imgs_list = self.imgs_dict[id_num]
        imgs = []  # 存储图像的张量
        for img_path, label in imgs_list:
            # --------------补0-----------------
            if img_path != "null":
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
            elif img_path == "null":
                img = torch.zeros([3, 224, 224])
            # -------------随机复制----------------
            # img = Image.open(img_path).convert('RGB')
            # if self.transform:
            #     img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs), id_num

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
                        id_num = int(image_name.split('groupid')[1].split('diff')[0])
                        if id_num not in data_info:
                            data_info[id_num] = []
                        data_info[id_num].append((img, label))
        self.imgs_dict = data_info
        print(f"Total data samples: {len(self.imgs_dict)}")

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs