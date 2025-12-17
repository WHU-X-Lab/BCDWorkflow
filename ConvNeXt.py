import timm
import torch
import torch.nn as nn

# class ConvNeXt(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 指定本地模型路径
#         local_model_path = './new_models/convnext_tiny.in12k_ft_in1k'
#         # 使用timm创建模型，并加载预训练权重
#         self.model = timm.create_model(
#             model_name='convnext_tiny.in12k_ft_in1k',
#             pretrained=False,
#             num_classes=2
#         )
#         # 然后手动加载权重，使用 strict=False
#         checkpoint = torch.load(local_model_path + '/pytorch_model.bin', map_location='cuda:0')
#
#         # 移除分类头的权重，因为它们尺寸不匹配
#         checkpoint = {k: v for k, v in checkpoint.items()
#                       if not k.startswith('head.') and not k.startswith('fc.')}
#
#         # 非严格模式加载，忽略不匹配的键
#         self.model.load_state_dict(checkpoint, strict=False)
#
#     def forward(self, x):
#         return self.model(x)

class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        local_model_path = './new_models/convnext_tiny.in12k_ft_in1k'
        # 创建模型但不包括最后的分类层
        self.model = timm.create_model(
            model_name='convnext_tiny.in12k_ft_in1k',
            pretrained=False,
            num_classes=0,  # 关键：设置为0，不要分类头
            global_pool=''   # 不进行全局池化
        )
        # 然后手动加载权重，使用 strict=False
        checkpoint = torch.load(local_model_path + '/pytorch_model.bin', map_location='cuda:0')
        # 移除分类头的权重，因为它们尺寸不匹配
        checkpoint = {k: v for k, v in checkpoint.items()
                      if not k.startswith('head.') and not k.startswith('fc.')}

        # 非严格模式加载，忽略不匹配的键
        self.model.load_state_dict(checkpoint, strict=False)

        # 获取特征维度
        self.feature_dim = self.model.num_features
        print(f"ConvNeXt特征维度: {self.feature_dim}")
        # 全局平均池化层（用于特征提取）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
         # 提取特征
        features = self.model(x)  # [batch, channels, height, width]
        # 直接返回特征向量
        features_pooled = self.global_pool(features)  # [batch, channels, 1, 1]
        features_flat = self.flatten(features_pooled)  # [batch, channels]
        return features_flat
