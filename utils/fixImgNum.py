import os
import shutil
from tqdm import tqdm

# 设置文件夹路径
folder_path = 'train_diff'
folder_0 = os.path.join(folder_path, '0')
# folder_1 = os.path.join(folder_path, '1')
folder_else = os.path.join(folder_path, 'else')

# 创建 else 文件夹（如果不存在）
if not os.path.exists(folder_else):
    os.makedirs(folder_else)

# 获取所有图像文件名
images_0 = [f for f in os.listdir(folder_0) if f.endswith('.jpg') or f.endswith('.png')]
# images_1 = [f for f in os.listdir(folder_1) if f.endswith('.jpg') or f.endswith('.png')]

# 处理 0 文件夹中的图像
for image in tqdm(images_0):
    # 获取图像 id
    image_id = image.split('group')[1].split('diff')[0]
    # 获取所有与该 id 相关的图像
    related_images = [f for f in images_0 if image_id == f.split('group')[1].split('diff')[0]]
    # 如果图像数量大于 4，则将多余的图像移动到 else 文件夹中
    if len(related_images) > 4:
        for i in range(4, len(related_images)):
            if not os.path.exists(os.path.join(folder_else,related_images[i])):
                shutil.move(os.path.join(folder_0, related_images[i]), folder_else)
    # 如果图像数量小于 4，则复制该 id 对应的图像，直到数量达到 4
    elif len(related_images) < 4:
        for i in range(len(related_images), 4):
            new_image_name = related_images[0].split('.')[0] + '_' + str(i) + '.' + related_images[0].split('.')[1]
            shutil.copy(os.path.join(folder_0, related_images[0]), os.path.join(folder_0, new_image_name))

# # 处理 1 文件夹中的图像（与上面类似）
# for image in tqdm(images_1):
#     image_id = image.split('groupid')[1].split('diff')[0]
#     related_images = [f for f in images_1 if image_id in f]
#     if len(related_images) > 4:
#         for i in range(4, len(related_images)):
#             if not os.path.exists(os.path.join(folder_else,related_images[i])):
#                 shutil.move(os.path.join(folder_1, related_images[i]), folder_else)
#     elif len(related_images) < 4:
#         for i in range(len(related_images), 4):
#             new_image_name = related_images[0].split('.')[0] + '_' + str(i) + '.' + related_images[0].split('.')[1]
#             shutil.copy(os.path.join(folder_1, related_images[0]), os.path.join(folder_1, new_image_name))
#
