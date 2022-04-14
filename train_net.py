import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import time
import random
import logging
import easydict
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as opj
from ptflops import get_model_complexity_info
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import pickle
import timm
import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, grad_scaler
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Warmup Learning rate scheduler
from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = timm.create_model('regnety_040', pretrained=True, drop_path_rate=0.2, num_classes=88)
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 88)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Network_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True,
                                         drop_path_rate=0,
                                         )

        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 88)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Train_Dataset(Dataset):
    def __init__(self, df, args, transform=None):
        self.root = args.root
        self.img_path = df['file_name'].values
        self.target = df['label_idx'].values
        self.transform = transform

        print(f'Dataset size:{len(self.img_path)}')

    def __getitem__(self, idx):
        image = cv2.imread(opj(self.root, 'train', self.img_path[idx])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image, target

    def __len__(self):
        return len(self.img_path)


class Test_dataset(Dataset):
    def __init__(self, df, args, transform=None):
        self.root = args.root
        self.test_img_path = df['file_name'].values
        self.transform = transform

        print(f'Test Dataset size:{len(self.test_img_path)}')

    def __getitem__(self, idx):
        image = cv2.imread(opj(self.root, 'test', self.test_img_path[idx])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image

    def __len__(self):
        return len(self.test_img_path)


# Logging
def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)


def get_train_augmentation(img_size, ver):
    if ver == 1:  # for validset
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    if ver == 2:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine((20)),
            transforms.RandomRotation(90),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform


def get_loader(df, args, phase: str, batch_size, shuffle,
               num_workers, transform):
    if phase == 'test':
        dataset = Test_dataset(df, args, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = Train_Dataset(df, args, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                 drop_last=False)
    return data_loader


class Trainer():
    def __init__(self, args, args2, save_path):
        '''
        args: arguments
        save_path: Model 가중치 저장 경로
        '''
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Logging
        log_file = os.path.join(save_path, 'log.log')
        self.logger = get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args2)
        # self.logger.info(args.tag)

        # Train, Valid Set load
        ############################################################################
        #with open('%s/data/train.pkl' % args.root, 'rb') as f:
        #    train_imgs = pickle.load(f)

        df_train = pd.read_csv("%s/train_df.csv" % args.root)
        train_labels = df_train["label"]

        label_unique = sorted(np.unique(train_labels))
        label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}

        train_labels = [label_unique[k] for k in train_labels]
        df_train['label_idx'] = train_labels
        # if args.image_type is not None:
        #     df_train['img_path'] = df_train['img_path'].apply(lambda x: x.replace('train_imgs', args.image_type))
        #     df_train['img_path'] = df_train['img_path'].apply(lambda x: x.replace('test_imgs', 'test_1024'))

        kf = StratifiedKFold(n_splits=args2.Kfold, shuffle=True, random_state=args2.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(df_train)), y=df_train['label'])):
            df_train.loc[val_idx, 'fold'] = fold
        val_idx = list(df_train[df_train['fold'] == int(args2.fold)].index)

        df_val = df_train[df_train['fold'] == args2.fold].reset_index(drop=True)
        df_train = df_train[df_train['fold'] != args2.fold].reset_index(drop=True)

        # Augmentation
        self.train_transform = get_train_augmentation(img_size=args2.img_size, ver=args2.aug_ver)
        self.test_transform = get_train_augmentation(img_size=args2.img_size, ver=1)

        # TrainLoader
        self.train_loader = get_loader(df_train, args, phase='train', batch_size=args2.batch_size, shuffle=True,
                                       num_workers=args2.num_workers, transform=self.train_transform)
        self.val_loader = get_loader(df_val, args, phase='train', batch_size=args2.batch_size, shuffle=False,
                                     num_workers=args2.num_workers, transform=self.test_transform)

        # Network
        self.model = Network().to(self.device)
        macs, params = get_model_complexity_info(self.model, (3, args2.img_size, args2.img_size), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.optimizer = optim.Lamb(self.model.parameters(), lr=args2.initial_lr, weight_decay=args2.weight_decay)

        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * args2.warm_epoch)

        if args2.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args2.milestone,
                                                                  gamma=args2.lr_factor, verbose=True)
        elif args2.scheduler == 'cos':
            tmax = args2.tmax  # half-cycle
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tmax, eta_min=args2.min_lr,
                                                                        verbose=True)
        elif args2.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args2.max_lr,
                                                                 steps_per_epoch=iter_per_epoch, epochs=args2.epochs)

        if args2.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Train / Validate
        best_loss = np.inf
        best_acc = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(1, args2.epochs + 1):
            self.epoch = epoch

            if args2.scheduler == 'cos':
                if epoch > args2.warm_epoch:
                    self.scheduler.step()

            # Training
            train_loss, train_acc, train_f1 = self.training(args2)

            # Model weight in Multi_GPU or Single GPU
            state_dict = self.model.module.state_dict() if args2.multi_gpu else self.model.state_dict()

            # Validation
            val_loss, val_acc, val_f1 = self.validate(args2, phase='val')

            # Save models
            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc
                best_f1 = val_f1

                torch.save({'epoch': epoch,
                            'state_dict': state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            }, os.path.join(save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args2.patience:
                break

        self.logger.info(
            f'\nBest Val Epoch:{best_epoch} | Val Loss:{best_loss:.4f} | Val Acc:{best_acc:.4f} | Val F1:{best_f1:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')

    # Training
    def training(self, args):
        self.model.train()
        train_loss = AvgMeter()
        train_acc = 0
        preds_list = []
        targets_list = []

        scaler = grad_scaler.GradScaler()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)

            if self.epoch <= args.warm_epoch:
                self.warmup_scheduler.step()

            self.model.zero_grad(set_to_none=True)
            if args.amp:
                with autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                scaler.scale(loss).backward()

                # Gradient Clipping
                if args.clipping is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)

                scaler.step(self.optimizer)
                scaler.update()

            else:
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
                self.optimizer.step()

            if args.scheduler == 'cycle':
                if self.epoch > args.warm_epoch:
                    self.scheduler.step()

            # Metric
            train_acc += (preds.argmax(dim=1) == targets).sum().item()
            preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
            targets_list.extend(targets.cpu().detach().numpy())
            # log
            train_loss.update(loss.item(), n=images.size(0))

        train_acc /= len(self.train_loader.dataset)
        train_f1 = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss.avg:.3f} | Acc:{train_acc:.4f} | F1:{train_f1:.4f}')
        return train_loss.avg, train_acc, train_f1

    # Validation or Dev
    def validate(self, args, phase='val'):
        self.model.eval()
        with torch.no_grad():
            val_loss = AvgMeter()
            val_acc = 0
            preds_list = []
            targets_list = []

            for i, (images, targets) in enumerate(self.val_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                preds = self.model(images)
                loss = self.criterion(preds, targets)

                # Metric
                val_acc += (preds.argmax(dim=1) == targets).sum().item()
                preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
                targets_list.extend(targets.cpu().detach().numpy())

                # log
                val_loss.update(loss.item(), n=images.size(0))
            val_acc /= len(self.val_loader.dataset)
            val_f1 = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

            self.logger.info(f'{phase} Loss:{val_loss.avg:.3f} | Acc:{val_acc:.4f} | F1:{val_f1:.4f}')
        return val_loss.avg, val_acc, val_f1