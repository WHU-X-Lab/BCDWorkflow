import torch
from torchview import draw_graph
from SimpleViT import SimpleViT


model = SimpleViT()
model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    device='cpu',
    expand_nested=True,
    depth=1,  # 控制展开深度
)
model_graph.visual_graph.render('model_arch')
# image = 'groupid54diff0x149521.jpg'
# image_id = image.split('groupid')[1].split('diff')[0]
# print("img_id",image_id)
# n=torch.FloatTensor(3,3,2,3).fill_(1)
# print("before:",n)
# # n[:,0:1,1:2]=0
# # n[:,1:,0:1]=0
# n = torch.mean(n.view(n.size(0), -1), dim=1)
# print("after:",n)
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print(f"CUDA 0 is available")
