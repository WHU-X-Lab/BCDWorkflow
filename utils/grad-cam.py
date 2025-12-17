import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json


# 图片预处理
def img_preprocess(img_in):
    # img = img_in.copy()						
    # img = img[:, :, ::-1]   				# 1
    # img = np.ascontiguousarray(img)			# 2
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.9101764, 0.9437504, 0.9431395], std=[0.15301877, 0.08691867, 0.11973674]),
    # ])
    # img = transform(img)
    # img = img.unsqueeze(0)					# 3
    # return img
    raw_image = cv2.resize(img_in, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.891972, 0.93623203, 0.9399001], std= [0.16588122, 0.090553254, 0.12211764]),
        ])(raw_image[..., ::-1].copy())
    image = torch.unsqueeze(image, 0).cuda()				# 3
    return image

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # cam_img = 0.3 * heatmap + 0.7 * img
    cam_img = 0.7 * heatmap + 0.3 * img
    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)

if __name__ == '__main__':
    path_img = r'predict_wrong\1\ex3topid100909168osmid19056.jpg'
    # json_path = './cam/labels.json'
    output_dir = './test_img'

    # with open(json_path, 'r') as load_f:
    #     load_json = json.load(load_f)
    # classes = {int(key): value for (key, value)
    #            in load_json.items()}
	
	# 只取标签名
    # classes = list(classes.get(key) for key in range(1000))
    classes = ('0','1')

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)
    img_input = img_preprocess(img)

    # 加载 squeezenet1_1 预训练模型
    # net = GoogleNet(num_class=N_FEATURES)
    net = AlexNet(num_classes=N_FEATURES,init_weights=True)
    pthfile = 'save_model/alexnet/last_model.pth'
    net.load_state_dict(torch.load(pthfile))
    net.cuda()
    net.eval()														# 8
    # print(net)

    # 注册hook
    # net.d4.register_forward_hook(farward_hook)	# 9
    # net.d4.register_backward_hook(backward_hook)
    net.features[10].register_forward_hook(farward_hook)	# 9
    net.features[10].register_backward_hook(backward_hook)
    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    # output【0，预测数值】
    class_loss = output[0,1]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, output_dir)

