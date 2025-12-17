import torch
import torch.nn as nn


class SimpleViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 num_classes=1,
                 embed_dim=128,
                 num_heads=2,
                 num_layers=2):
        super().__init__()

        # 图像分块与嵌入
        # 输入图像的大小是224x224，分块大小为16，因此每个维度有224/16=14个块，总共有14x14=196个块
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        # 卷积层self.patch_embed将输入图像转换为形状为[batch_size, embed_dim, 14, 14]
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(self.num_patches + 1,1 , embed_dim)) #[197,1,128]

        # Transformer编码层
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 分类头
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, **kwargs):
        # 分块嵌入 [batch, 3, 224, 224] -> [batch=64, embed_dim=128, 14, 14]
        x = self.patch_embed(x)
        x = x.flatten(2).permute(2, 0, 1)  # x:[num_patches=196, batch=64, embed_dim=128]

        # 添加[CLS] token，长度变为197
        cls_tokens = self.cls_token.expand(-1, x.shape[1], -1)# [1,64,128]
        x = torch.cat([cls_tokens, x], dim=0) #[197,64,128]

        # 添加位置编码
        # self.pos_embed形状为 [197, 1, 128]，广播后为 [197, 64, 128]
        x = x + self.pos_embed

        # Transformer编码（输入维度 [197, 64, 128],启用 batch_first）
        x = x.permute(1, 0, 2)  # [batch, 197, 128]
        x = self.transformer(x)

        # 取 [CLS] token 输出 [64, 128]
        cls_output = x[:, 0, :]
        output = self.head(cls_output)#[64, 128]→[64,1]
        return output