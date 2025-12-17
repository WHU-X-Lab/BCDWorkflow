# -- coding:utf-8
import pathlib
from PIL import Image
from torch.utils.data import Dataset
import os


class BuildingDataset(Dataset):
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
            for img_path1 in img_paths1:
                class_id = 0
                if img_path == '0':
                    class_id = 0
                elif img_path == '1':
                    class_id = 1
                self.classes_for_all_imgs.append(class_id)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.data_info):
            raise IndexError(f"Index {index} out of range for data_info of length {len(self.data_info)}")
        path_img, label = self.data_info[index]
        # 打开图片，默认为PIL，需要转成RGB
        img = Image.open(path_img).convert('RGB')
        # 如果预处理的条件不为空，应该进行预处理操作
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

        # 自定义方法，用于返回所有图片的路径以及标签

    def get_img_info(self, data_dir):
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        data_info = []
        for sub_dir in data_dir.iterdir():
            if sub_dir.is_dir():
                for img in sub_dir.iterdir():
                    if img.suffix == '.jpg':
                        label = int(sub_dir.name) if sub_dir.name.isdigit() else -1
                        data_info.append((img, label))
        self.data_info = data_info
        print(f"Total data samples: {len(self.data_info)}")

    def get_classes_for_all_imgs(self):
        # print(f"Total classes returned: {len(self.classes_for_all_imgs)}")
        return self.classes_for_all_imgs
