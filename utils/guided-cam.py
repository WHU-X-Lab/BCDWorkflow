from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from config import *
from GoogleNet import GoogLeNet
from model import AlexNet

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.size()[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet(gcam)[..., :3] * 255.0
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        # gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        gcam = cmap.astype(np.float64) * 0.3 + raw_image.astype(np.float64) *0.7
    gcam = np.uint8(gcam)
    im=Image.fromarray(gcam) 
    im.save(filename)
    # cv2.imwrite(img = gcam,filename = filename)


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

image_dir = "predict_area"
dirs = []
for dir in os.listdir(image_dir):
    dirs.append(dir)
dirs.sort()


# pthfile = r'use_model\alex_rect_last\last_model.pth'
# net1 = GoogleNet(num_class=N_FEATURES)
# net1 = AlexNet(num_classes=2,init_weights=True)
# net1.load_state_dict(torch.load(pthfile))
# net1.cuda()
# net1.eval()	
# print(net1)

net1 = GoogLeNet(num_classes=2,init_weights=True,aux_logits=True)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    net1 = nn.DataParallel(net1)
net1.to(device)
# # net1.load_state_dict(torch.load('D:\变化探测研究\训练好的模型\\202112222100\last_model.pth'))
net1.load_state_dict(torch.load('use_model\\google_area_best\\best_model.pth'))
net1.cuda()
net1.eval()
print(net1)

bp = BackPropagation(model=net1)
gcam = GradCAM(model=net1)

global_id = 0
label = 0  # label是图像的ground-truth label
for dir in dirs:
    cur_dir = os.path.join(image_dir, dir)
    label = label + 1
    for jpeg in os.listdir(cur_dir):
        raw_image = cv2.imread(os.path.join(cur_dir, jpeg))
        raw_image = cv2.resize(raw_image, (224,) * 2)
        image = transforms.Compose(
            [
                transforms.ToTensor(),
                # valid_test2
                transforms.Normalize(mean = [0.8904361, 0.9363585, 0.94014686], std = [0.165799, 0.09002642, 0.12338792])
                # compareValid
                # transforms.Normalize(mean = [0.84808147, 0.9096524, 0.91053045], std = [0.20253241, 0.10373465, 0.14877456])
            ])(raw_image[..., ::-1].copy())
        image = torch.unsqueeze(image, 0).cuda()

        probs, ids = bp.forward(image)
        _ = gcam.forward(image)
        # ids[:, [0]]这里0表示top-1的预测类别
        gcam.backward(ids=ids[:, [1]])
        # regions = gcam.generate(target_layer='features.10')
        regions = gcam.generate(target_layer='module.inception5b.branch3.1.conv')
        save_gradcam(
        filename=osp.join(
            'cam', jpeg),
            gcam=regions[0, 0],
            raw_image=raw_image,
        )

        # global_id += 1
