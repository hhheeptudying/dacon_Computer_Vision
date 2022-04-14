import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

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

def get_data(args):
    train = pd.read_csv(args["TRAIN_LABEL_CSV"])
    train_list = glob.glob(args["TRAIN_PATH"] + "/**/*")
    test_list = glob.glob(args["TEST_PATH"] + "/*")

    path_df = pd.DataFrame({"path" : train_list})
    train["path"] = path_df["path"]
    train["path"] = train["path"].astype(np.str)
    train_df = train
    le = LabelEncoder()
    result = le.fit_transform(train_df['label'])
    train_df['label_num'] = result.astype(np.str)



    print(f"Train 데이터 수 : {len(train_list)}\nTest 데이터 수 : {len(test_list)}\n")
    print(f"Target 데이터 수 : \n{train_df['file_name'].value_counts()}\n")
    
    return train_df, train_list, test_list


def to_categorical(y, num_classes=88):
    ''' 
        change str to int to categorical
        1-hot encodes a tensor
                            '''
    return np.eye(num_classes)[y]
    
class InvasiveDataset(Dataset):
    def __init__(self, dataframe,transform=None):
        super().__init__()
        self.df = dataframe
        self.file_list = dataframe["path"].values
        self.label = dataframe["label_num"].values
        self.transform = transform
       

    def __getitem__(self, index):
        image = self.file_list[index]
        label = self.label[index]# label in Series. Change to array.

        image = Image.open(image).convert('RGB')

        label = to_categorical(int(label))
        if self.transform:
            image = self.transform(image)
        
        return image, label
        
    def __len__(self):
        return len(self.file_list)

def get_train_transform(args):
    return transforms.Compose([
        transforms.RandomResizedCrop(args["RESIZE"], scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()    ])

def get_valid_transform(args):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args["RESIZE"]),
        transforms.ToTensor()    ])

def get_data_loader(train_df, valid_df=None):
    
    if valid_df is not None: # k-fold
        train_set = InvasiveDataset(dataframe=train_df, transform=get_train_transform(args))
        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=1)
        valid_set = InvasiveDataset(valid_df, transform=get_valid_transform(args))
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)     
    else:
        train_, valid_ = train_test_split(train_df, test_size=0.1, random_state=42)
        train_set = InvasiveDataset(train_, transform=get_train_transform(args))
        train_loader = DataLoader(train_set, batch_size=args["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=1)
        valid_set = InvasiveDataset(valid_, transform=get_valid_transform(args))
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)
    
    return train_loader, valid_loader