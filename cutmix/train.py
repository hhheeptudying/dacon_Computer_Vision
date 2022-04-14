import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from pylab import rcParams
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import dataloader
from torchvision import datasets

from data_load import InvasiveDataset, get_data, get_train_transform, get_valid_transform, get_data_loader
from get_model import get_model,check_batch ,fold_train

from cutmix import rand_bbox, cutmix_plot

args = {"TRAIN_LABEL_CSV" : '/data/dacon_Computer_Vision_data/train_df.csv',
        "TRAIN_PATH" : "/data/dacon_Computer_Vision_data/train",
        "TEST_PATH" : "/data/dacon_Computer_Vision_data/test",
        "RESIZE" : 512,#(224,224),
        "LEARNING_RATE" : 0.001,
        "WEIGHT_DECAY" : 0.003,
        "BATCH_SIZE" : 32,
        "FEATURE_EXTRACTING" : True,
        "NUM_EPOCHS" : 50,
        "MEAN" : (0.485, 0.456, 0.406),
        "STD" : (0.229, 0.224, 0.225),
        "BETA" : 1.0,
        "MODEL" : "efficient",
        "MODEL_PATH" : ".",
        "NUM_FOLDS" : 5,
        "DEVICE" : torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')}

#data
train_df, train_list, test_list = get_data(args)
train_loader, valid_loader = get_data_loader(train_df)

#model_check
model, criterion, optimizer = get_model(args)
check_batch(model, train_loader=train_loader, criterion=criterion, optimizer=optimizer,args=args)

#cutmix_check
cutmix_plot(train_loader)

#5-Fold Train
fold_train(args, train_df)

