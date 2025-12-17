#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration of Googlenet
"""

# configuration of alexnet
weight_path = "save_model/area_Goo/best_model.pth"  # path to the weigths
alexnet_path = "save_model/area_Goo/best_model.pth"  # path to the net
N_FEATURES = 1

# params of training
MAX_EPOCH = 150
BATCH_SIZE = 16
LR = 0.00001
device = 'cuda:0'

# configuration of decission tree
MAX_DEPTH = 5
tree_path = 'dt_alex.pkl'

# data path
## the path of training data
train_dir = "train_area"
valid_dir = "valid_area"
# train_dir = "origin_area"
# valid_dir = "valid_area"
## valid_dir=pathlib.Path("valid")
# SGD²ÎÊý
weight_decay = 0.00005
# milestones = [7,28,70,150]
gamma = 0.1
weight_decay_f = 0.00001
milestones = [30,50,70,90]
## the path of data for prediction
pred_dir = 'predict'