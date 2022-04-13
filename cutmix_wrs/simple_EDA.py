import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# Python
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import random
import glob
from collections import Counter
pd.set_option('display.max_columns', None)

# Visualization
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
sns.set(style='whitegrid')

# Image albumentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score,f1_score

# Pytorch for Deep Learning
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import amp

#GPU
from accelerate import Accelerator
accelerator = Accelerator()

params = {
    'seed': 42,
    'model': 'swin_small_patch4_window7_224',
    'size': 224,
    'inp_chennels': 3,
    'device': accelerator.device, # device(type='cuda')
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'batch_size': 32,
    'num_workers': 0,
    'epochs': 5,
    'out_features': 1,
    'name': 'CosineAnnealingLR',
    'T_max': 10,
    'min_lr': 1e-6,
    'num_tta': 1
}

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(params['seed'])

# Weights and Biases Tool
import wandb

train_dir = ('/data/Computer_Vision/train')
test_dir = ('/data/Computer_Vision/test')
train_df = pd.read_csv('/data/Computer_Vision/train_df.csv')
test_df = pd.read_csv('/data/Computer_Vision/test_df.csv')

def return_filepath(name, folder=train_dir):
    path = os.path.join(folder, f'{name}')
    return path

train_df['image_path'] = train_df['file_name'].apply(lambda x: return_filepath(x))
test_df['image_path'] = test_df['file_name'].apply(lambda x: return_filepath(x, folder=test_dir))
print(train_df.head())

def get_train_transforms():
    return A.Compose(
        [
            A.Resize(params['size'], params['size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.5),
            A.RandomBrightness(limit=0.6, p=0.5),
            A.Cutout(
                num_holes=10, max_h_size=12, max_w_size=12,
                fill_value=0, always_apply=False, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(params['size'],params['size']),
            ToTensorV2(p=1.0)
        ]
    )

def get_test_transforms():
    return A.Compose(
        [
            A.Resize(params['size'],params['size']),
            ToTensorV2(p=1.0)
        ]
    )


class SETIDataset(Dataset):
    def __init__(self, images_filepaths, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.targets = targets
        self.transform=transform
        
    def __len__(self):
        return len(self.images_filepaths)
    
    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        # image = np.load(image_filepath).astype(np.float32)  # npz 파일 load
        # image = np.vstack(image).transpose((1, 0))  #ex  (6, 273, 256) -> (1638, 256) -> (256, 1638)
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
            
        label = torch.tensor(self.targets[idx]).float()
        return image, label
labels = train_df['label']

label_unique = sorted(np.unique(labels))
label_unique = {key:value for key, value in zip(label_unique,range(len(label_unique)))}

label = [label_unique[k] for k in labels]
(X_train, X_valid, y_train, y_valid) = train_test_split(train_df['image_path'], label, test_size=0.2,
                stratify=train_df['label'], shuffle=True, random_state=params['seed'])

train_dataset = SETIDataset(
    images_filepaths=X_train.values,
    targets=y_train,
    transform=get_train_transforms()
)

valid_dataset = SETIDataset(
    images_filepaths=X_valid.values,
    targets=y_valid,
    transform=get_valid_transforms()
)

def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # 중심좌표
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # corner 좌표
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def shuffle_minibatch(x, y):
    assert x.size(0)== y.size(0)
    indices = torch.randperm(x.size(0))
    return x[indices], y[indices]

def cutmix(x, y, alpha=1.0):
    if alpha > 0:
        lam= np.random.beta(alpha, alpha)  # 0~1
    else:
        lam=1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(params['device'])# 배치내의 다른 이미지와 mixup하기 위해
    index = index.float()
    x_train_shuffled, y_train_shuffled = shuffle_minibatch(x, y_train)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size()[2], x.size()[3], lam) # (32, 1, 224, 224)

    x[:, bbx1:bbx2, bby1:bby2] = x[:,bbx1:bbx2, bby1:bby2]  # 원래 이미지에 배치내의 다른 이미지를 삽입
    y_a, y_b = y, y[index[i]]  # 이제 이미지당 정답은 2개
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
def use_roc_score(output, target):
    try:
        y_pred = torch.sigmoid(output).cpu()
        y_pred = y_pred.detach().numpy()
        target = target.cpu()
        return f1_score(target, y_pred,average='macro')
    except:
        return 0.5
class_counts = []
for i in range(len(label_unique)):
    count = sorted(Counter(y_train).most_common())[i][1]
    class_counts.append(count)
num_samples = sum(class_counts) 
labels = y_train

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

train_loader = DataLoader(
    train_dataset, batch_size=params['batch_size'], sampler = sampler, # sampler로 균형맞춤
    num_workers=params['num_workers'], pin_memory=True)

val_loader = DataLoader(
    valid_dataset, batch_size=params['batch_size'], shuffle=False,
    num_workers=params['num_workers'], pin_memory=True)

print(train_loader)

# timm에 있는 swin model 종류
timm.list_models('swin*')

class SwinNet(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['out_features'],
                inp_channels=params['inp_chennels'], pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                      in_chans=inp_channels)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, out_features, bias=True)
    def forward(self, x):
        return self.model(x)
    
model = SwinNet()
model = model.to(params['device'])
criterion = nn.BCEWithLogitsLoss().to(params['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'],
                            weight_decay=params['weight_decay'],
                            amsgrad=False)

scheduler = CosineAnnealingLR(optimizer, T_max=params['T_max'],
                             eta_min=params['min_lr'],
                             last_epoch=-1)

def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    scaler = amp.GradScaler()
    
    for i , (images, target) in enumerate(stream, start=1):
        images = images.to(params['device'])
        target = target.to(params['device']).float().view(-1, 1) # (32x1)
        images, targets_a, targets_b, lam = cutmix(images, target.view(-1, 1))
        
        with amp.autocast(enabled=True):
            output = model(images)
            loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)
            
        accelerator.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        roc_score = use_roc_score(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('ROC', roc_score)
        wandb.log({"Train Epoch":epoch,"Train loss": loss.item(), "Train ROC":roc_score})

        stream.set_description( # tqdm
        "Epoch: {epoch}. Train.      {metric_monitor}".format(
            epoch=epoch,
            metric_monitor=metric_monitor)
    )
        
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            roc_score = use_roc_score(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('ROC', roc_score)
            wandb.log({"Valid Epoch": epoch, "Valid loss": loss.item(), "Valid ROC":roc_score})
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
            
            targets = target.detach().cpu().numpy().tolist()
            outputs = output.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets

best_roc = -np.inf
best_epoch = -np.inf
best_model_name = None
    
for epoch in range(1, params['epochs']+1):
    
    run = wandb.init(project='Seti-Swin',
                    config=params,
                    job_type='train',
                    name=f'Swin Transformer_epoch{epoch}')
    
    train(train_loader, model, criterion, optimizer, epoch, params)
    predictions, valid_targets = validate(val_loader, model, criterion, epoch, params)
    roc_auc = round(roc_auc_score(valid_targets, predictions), 3)
    torch.save(model.state_dict(),f"{params['model']}_{epoch}_epoch_{roc_auc}_roc_auc.pth")
    
    if roc_auc > best_roc:
        best_roc = roc_auc
        best_epoch = epoch
        best_model_name = f"{params['model']}_{epoch}_epoch_{roc_auc}_roc_auc.pth"
        
    scheduler.step()

    print(f'The best ROC: {best_roc} was achieved on epoch: {best_epoch}.')
    print(f'The Best saved model is: {best_model_name}')