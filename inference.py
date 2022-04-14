import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time
from efficientnet_pytorch import EfficientNet


device = torch.device('cuda')
train_y = pd.read_csv("/data/dacon_Computer_Vision_data/train_df.csv")
test_png = sorted(glob('/data/dacon_Computer_Vision_data/test/*.png'))

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img
#label 설정
train_labels = train_y["label"]
label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

test_imgs = [img_load(n) for n in tqdm(test_png)]
#model_load
model = EfficientNet.from_pretrained('efficientnet-b5',num_classes = 88)

model.fc = nn.Linear(1000,88)
model = model.to(device)

model.load_state_dict(torch.load('/data/dacon_Computer_Vision_data/model/Mixup_EFFb3.pt'))


model.eval()
f_pred = []

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode=='train':
            augmentation = random.randint(0,2)
            if augmentation==1:
                img = img[::-1].copy()
            elif augmentation==2:
                img = img[:,::-1].copy()
        img = transforms.ToTensor()(img)
        if self.mode=='test':
            pass
        
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model(x)
        return x

test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=4)




with torch.no_grad():
    for batch in (test_loader):
        x = torch.tensor(batch[0], dtype = torch.float32, device = device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]

submission = pd.read_csv("/data/dacon_Computer_Vision_data/sample_submission.csv")

submission["label"] = f_result

submission.to_csv("mixup_Effb5.csv", index = False)