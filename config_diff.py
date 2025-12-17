# -- coding:utf-8
"""Configuration of Diffnet
"""

# configuration of different
weight_path = "save_model/different/best_model.pth"  # path to the weigths
alexnet_path = "save_model/different/best_model.pth"  # path to the net
N_FEATURES = 2 #简单模型改为1

# params of training
MAX_EPOCH = 100
BATCH_SIZE = 64

LR = 0.0001
device = 'cuda:0'

# configuration of decission tree
MAX_DEPTH = 5
tree_path = 'dt.pkl'

# data path
## the path of training data
train_dir = "train_diff"
valid_dir = "valid_diff"


# SGD参数
gamma = 0.1
# weight_decay_f = 0.00001
weight_decay_f = 0.0001#增大L2权重衰减，正则化增强
milestones = [50,70,90,110]
## the path of data for prediction
pred_dir = 'valid_diff'
