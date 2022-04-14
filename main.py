import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import pickle
import os
import timm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time
import argparse
from train_net import Trainer, Test_dataset, Network, Network_test
import easydict
from os.path import join as opj

device = torch.device('cuda:0')

args2 = easydict.EasyDict(
    {'exp_num': '0',

     # Path settings
     'data_path': '../data',
     'Kfold': 5,
     'model_path': 'model/regnety_040',
     'image_type': 'train',

     # Model parameter settings
     'encoder_name': 'regnety_040',
     'drop_path_rate': 0.2,

     # Training parameter settings
     ## Base Parameter
     'img_size': 288,
     'batch_size': 60,
     'epochs': 60,
     'optimizer': 'Lamb',
     'initial_lr': 5e-6,
     'weight_decay': 1e-3,

     ## Augmentation
     'aug_ver': 2,

     ## Scheduler (OnecycleLR)
     'scheduler': 'cycle',
     'warm_epoch': 5,
     'max_lr': 1e-3,

     ### Cosine Annealing
     'min_lr': 5e-6,
     'tmax': 145,

     ## etc.
     'patience': 15,
     'clipping': None,

     # Hardware settings
     'amp': True,
     'multi_gpu': False,
     'logging': False,
     'num_workers': 4,
     'seed': 42
     })


def get_parser():
    parser = argparse.ArgumentParser('Dacon Anomaly Detection', add_help=False)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--root', default='/mnt/dms/PMH/dacon_anomalib', type=str)
    parser.add_argument('--train', action='store_true', default=False, help='if you want to train model')
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    return args

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode

        print(f'Dataset size:{len(self.img_paths)}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == 'train':
            augmentation = random.randint(0, 2)
            if augmentation == 1:
                img = img[::-1].copy()
            elif augmentation == 2:
                img = img[:, ::-1].copy()
        img = transforms.ToTensor()(img)
        if self.mode == 'test':
            pass

        label = self.labels[idx]
        return img, label


## Train Model
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


def predict(encoder_name, test_loader, device, model_path):
    model = Network_test(encoder_name).to(device)
    model.load_state_dict(torch.load(opj('/mnt/dms/PMH/dacon_anomalib/result', model_path, 'best_model.pth'))['state_dict'])
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
            preds = model(images)
            preds = torch.softmax(preds, dim=1)
            preds_list.extend(preds.cpu().tolist())

    return np.array(preds_list)


def ensemble_5fold(model_path_list, test_loader, device):
    predict_list = []
    for model_path in model_path_list:
        prediction = predict(encoder_name='regnety_040', test_loader=test_loader, device=device, model_path=model_path)
        predict_list.append(prediction)
    ensemble = (predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)

    return ensemble


def make_pseudo_df(train_df, test_df, ensemble, step, threshold=0.9, z_sample=500):
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    test_df_copy['label'] = np.nan
    test_df_copy['label_idx'] = ensemble.argmax(axis=1)
    pseudo_test_df = test_df_copy.iloc[np.where(ensemble > threshold)[0]].reset_index(drop=True)
    z_idx  = pseudo_test_df[pseudo_test_df['label_idx'] == 0].sample(n=z_sample, random_state=42).index.tolist()
    ot_idx = pseudo_test_df[pseudo_test_df['label_idx'].isin([*range(1, 8)])].index.tolist()
    pseudo_test_df = pseudo_test_df.iloc[z_idx + ot_idx]

    train_df_copy = train_df_copy.append(pseudo_test_df, ignore_index=True).reset_index(drop=True)  # reset_index
    # print(f'Make train_{step}step.csv')
    train_df_copy.to_csv(f'/mnt/dms/PMH/dacon_anomalib/model/regnety_040/train_{step}step.csv', index=False)


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


def train(args, args2):
    print('<---- Training Params ---->')

    # Random Seed
    seed = args2.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.root, args2.model_path, (args2.exp_num).zfill(3))

    # Create model directory
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, args2, save_path)

    return save_path

    # epochs = args.epochs
    # model = Network().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    # scaler = torch.cuda.amp.GradScaler()
    #
    # best_score = 0
    # for epoch in range(epochs):
    #     start = time.time()
    #     train_loss = 0
    #     train_pred = []
    #     train_y = []
    #     model.train()
    #     for batch in (train_loader):
    #         optimizer.zero_grad()
    #         x = torch.tensor(batch[0], dtype=torch.float32, device=device)
    #         y = torch.tensor(batch[1], dtype=torch.long, device=device)
    #         with torch.cuda.amp.autocast():
    #             pred = model(x)
    #         loss = criterion(pred, y)
    #
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #
    #         train_loss += loss.item() / len(train_loader)
    #         train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
    #         train_y += y.detach().cpu().numpy().tolist()
    #
    #     train_f1 = score_function(train_y, train_pred)
    #
    #     TIME = time.time() - start
    #     print(f'epoch : {epoch + 1}/{epochs}    time : {TIME:.0f}s/{TIME * (epochs - epoch - 1):.0f}s')
    #     print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    #
    #     if train_f1 > best_score or epoch % 10 == 0:
    #         best_score = train_f1
    #         torch.save(model.state_dict(), '%s/model/regnety_040/%d_epoch_model.pth' % (root, epoch))


def test(root, test_loader, label_unique):
    model = Network_test('regnety_040').to(device)
    model.load_state_dict(torch.load('%s/results/001/best_model.pth' % root))
    model.eval()
    f_pred = []

    with torch.no_grad():
        for batch in (test_loader):
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            with torch.cuda.amp.autocast():
                pred = model(x)
            f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

    label_decoder = {val: key for key, val in label_unique.items()}

    f_result = [label_decoder[result] for result in f_pred]

    ## 제출물 생성
    submission = pd.read_csv("%s/sample_submission.csv" % root)

    submission["label"] = f_result

    submission.to_csv("%s/sample_submission.csv" % root, index=False)


def main(args):

    img_size = 288
    root = args.root
    sub = pd.read_csv('%s/sample_submission.csv' % root)
    batch_size = args.batch_size

    df_train = pd.read_csv("%s/train_df.csv" % args.root)
    df_test = pd.read_csv("%s/test_df.csv" % args.root)
    test_transform = get_train_augmentation(img_size=img_size, ver=1)
    test_dataset = Test_dataset(df_test, args, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    train_labels = df_train["label"]

    label_unique = sorted(np.unique(train_labels))
    label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}

    train_labels = [label_unique[k] for k in train_labels]

    start = 0  # first time : Only Trainset
    steps = 6  # Number of pseudo labeling times
    for step in range(start, steps + 1):
        models_path = []
        args2.step = step
        for s_fold in range(5):  # 5fold
            args2.fold = s_fold
            args2.exp_num = str(s_fold)
            save_path = train(args, args2)
            models_path.append(save_path)
        ensemble = ensemble_5fold(models_path, test_loader, device)
        make_pseudo_df(df_train, df_test, ensemble, step + 1)

    # For submission
    sub.iloc[:, 1] = ensemble.argmax(axis=1)
    sub.to_csv(f'./submission.csv', index=False)

    '''
    if args.train:
        #train_imgs = [img_load(m) for m in tqdm(train_png)]
        Trainer(args, args2, root)
        with open('%s/data/train.pkl' % root, 'rb') as f:
            train_imgs = pickle.load(f)
        train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        train(args, root, train_loader)

    elif args.test:
        #test_imgs = [img_load(n) for n in tqdm(test_png)]
        with open('%s/data/test.pkl' % root, 'rb') as f:
            test_imgs = pickle.load(f)
        test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"] * len(test_imgs)), mode='test')
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
        test(root, test_loader, label_unique)
    '''

'''
# Dataset을 pkl 파일로 저장
#with open('/mnt/dms/PMH/dacon_anomalib/data/train.pkl', 'wb') as f:
#    pickle.dump(train_imgs, f)
'''

if __name__ == '__main__':
    args = get_parser()
    if args.test:
        df_train = pd.read_csv("%s/train_df.csv" % args.root)
        train_labels = df_train["label"]
        label_unique = sorted(np.unique(train_labels))
        label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}
        with open('%s/data/test.pkl' % args.root, 'rb') as f:
            test_imgs = pickle.load(f)
        test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"] * len(test_imgs)), mode='test')
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
        test(args.root, test_loader, label_unique)
    main(args)