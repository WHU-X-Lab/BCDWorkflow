import timm
import torch
import torch.nn as nn

# class SwinTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 指定本地模型路径
#         local_model_path = './new_models/swin_tiny_patch4_window7_224.ms_in1k'
#
#         # 使用timm创建模型，并加载预训练权重
#         self.model = timm.create_model(
#             model_name='swin_tiny_patch4_window7_224.ms_in1k',
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


#-----------------------Feature extract---------------------------

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
       # 指定本地模型路径
        local_model_path = './new_models/swin_tiny_patch4_window7_224.ms_in1k'

        # 使用timm创建模型，并加载预训练权重
        self.model = timm.create_model(
            model_name='swin_tiny_patch4_window7_224.ms_in1k',
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
        print(f"Swin Transformer特征维度: {self.feature_dim}")
        # Swin的输出已经是序列形式，需要取平均
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 提取特征（Swin输出的是序列）
        features = self.model(x)  # [batch, num_tokens, feature_dim]
        # 直接返回特征向量（对序列维度取平均）
        features_pooled = features.mean(dim=1)  # [batch, feature_dim]
        return features_pooled