import warnings

warnings.filterwarnings("ignore")
import datetime
import os
import random
import time
from glob import glob
from time import localtime, strftime

import cv2
import numpy as np
import timm
import torch
import torchvision.models
import torchvision.transforms
from albumentations import (
    CLAHE,
    Blur,
    CenterCrop,
    CoarseDropout,
    Compose,
    Cutout,
    Flip,
    GaussNoise,
    GridDistortion,
    HorizontalFlip,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    IAAEmboss,
    IAAPerspective,
    IAAPiecewiseAffine,
    IAASharpen,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    OpticalDistortion,
    RandomBrightnessContrast,
    RandomResizedCrop,
    RandomRotate90,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score
from torch import nn, normal
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, transforms, mode="train"):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == "train":
            augmentation = random.randint(0, 2)
            if augmentation == 1:
                img = img[::-1].copy()
            elif augmentation == 2:
                img = img[:, ::-1].copy()
        # img = transforms.ToTensor()(img)
        img = self.transforms(image=img)["image"]
        if self.mode == "test":
            pass

        label = self.labels[idx]
        return img, label


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=88
        )

    def forward(self, x):
        x = self.model(x)
        return x


def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


def img_load(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = cv2.resize(img, (700, 700))
    return img


def normalize_train_dataset(train_imgs):
    # To normalize the dataset, calculate the mean and std
    train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_imgs]
    train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_imgs]

    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])

    return [train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]


def normalize_test_dataset(test_imgs):
    test_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in test_imgs]
    test_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in test_imgs]

    test_meanR = np.mean([m[0] for m in test_meanRGB])
    test_meanG = np.mean([m[1] for m in test_meanRGB])
    test_meanB = np.mean([m[2] for m in test_meanRGB])

    test_stdR = np.mean([s[0] for s in test_stdRGB])
    test_stdG = np.mean([s[1] for s in test_stdRGB])
    test_stdB = np.mean([s[2] for s in test_stdRGB])

    return [test_meanR, test_meanG, test_meanB], [test_stdR, test_stdG, test_stdB]


def train_Transform(train_imgs):
    mean, std = normalize_train_dataset(train_imgs)
    return Compose(
        [
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            Resize(700, 700),
            Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def test_Transform(test_imgs):
    mean, std = normalize_test_dataset(test_imgs)
    return Compose(
        [
            Resize(700, 700),
            Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_test_data(train_data, test_data):
    train_imgs = [img_load(m[0]) for m in tqdm(train_data.imgs)]
    test_imgs = [img_load(n) for n in tqdm(test_data)]
    train_labels = list(train_data.targets)
    return train_imgs, test_imgs, train_labels


def train_test_dataloader(data, batch_size, mode="train"):
    if mode == "train":
        train_dataset = Custom_dataset(
            np.array(data),
            np.array(train_labels),
            transforms=train_Transform(train_imgs),
            mode="train",
        )
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        return train_loader
    else:
        test_dataset = Custom_dataset(
            np.array(data),
            np.array(["tmp"] * len(data)),
            transforms=test_Transform(data),
            mode="test",
        )
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
        return test_loader


def run_train(train_loader, epochs):
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    print(f"-----------------------Start Training-----------------------")
    best_score = 0
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        train_pred = []
        train_y = []
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                pred = model(x)
            loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() / len(train_loader)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_y += y.detach().cpu().numpy().tolist()

        train_f1 = score_function(train_y, train_pred)

        sec = time.time() - start
        tm = localtime(time.time())
        time_result = str(datetime.timedelta(seconds=sec)).split(".")
        print(
            f"epoch : {epoch+1}/{epochs}   train time : {time_result[0]}s   current time : {strftime('%Y-%m-%d %p %I:%M:%S', tm)}"
        )
        print(f"TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}")

        if epoch >= 10 and train_f1 >= best_score:
            torch.save(
                model.state_dict(),
                "open/aug_Effi_b4_1024" + f"/best_f1_epoch{epoch+1}.pth",
            )
            best_score = train_f1
            print(f"model is saved when epoch is : {epoch+1}")


if __name__ == "__main__":
    device = torch.device("cuda")
    train_data = torchvision.datasets.ImageFolder(root="open/split")
    test_data = sorted(glob("open/test/*.png"))
    # load pickle data
    #with open("train_data", "rb") as file:
    #    train_imgs = pickle.load(file)
        
    train_imgs, test_imgs, train_labels = train_test_data(train_data, test_data)
    train_loader = train_test_dataloader(train_imgs, batch_size=8, mode="train")
    seed_everything()
    run_train(train_loader, epochs=100)