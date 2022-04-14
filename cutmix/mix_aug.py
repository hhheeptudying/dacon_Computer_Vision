import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.transforms as transforms

def plot_train_transform(train_list, args, num_images=3, fig_size=(20,15)):
    
    preprocess1 = transforms.Compose([
        transforms.RandomResizedCrop(args["RESIZE"], scale=(0.5, 1.0)),
    ])

    preprocess2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
    ])    

    fig , axes = plt.subplots(num_images,4)
    fig.set_size_inches(fig_size)

    random_path = random.sample(train_list,num_images)


    for idx in range(num_images):
        img = Image.open(random_path[idx])
        img_origin = img.copy()
        img_aug1 = preprocess1(img)
        img_aug2 = preprocess2(img)
        img_aug3 = preprocess2(img_aug1)
        imgs = [img_origin, img_aug1, img_aug2, img_aug3]
        preprocess_name = ["Original Image", f"RandomResizedCrop(Resize:{args['RESIZE']})", "RandomHorizontalFlip"]
        for i in range(len(imgs)):

            axes[idx, i].imshow(imgs[i])
            if i ==len(imgs)-1:
                axes[idx, i].set_title(f"{preprocess_name[1]}+{preprocess_name[2]}\nSize : {imgs[i].size}")
            else:
                axes[idx, i].set_title(f"{preprocess_name[i]}\nSize : {imgs[i].size}")
            axes[idx, i].axis('off')
            
def plot_valid_transform(train_list, args, num_images=3, fig_size=(20,15)):
    
    preprocess1= transforms.Compose([
        transforms.Resize(256),
    ])

    preprocess2 = transforms.Compose([
            transforms.CenterCrop(args["RESIZE"]),
    ])
    
    fig , axes = plt.subplots(num_images,4)
    fig.set_size_inches(fig_size)

    random_path = random.sample(train_list,num_images)

    for idx in range(num_images):
        img = Image.open(random_path[idx])
        img_origin = img.copy()
        img_aug1 = preprocess1(img)
        img_aug2 = preprocess2(img)
        img_aug3 = preprocess2(img_aug1)
        imgs = [img_origin, img_aug1, img_aug2, img_aug3]
        preprocess_name = ["Original Image", "Resize(256)", f"CenterCrop(Resize:{args['RESIZE']})"]
        for i in range(len(imgs)):

            axes[idx, i].imshow(imgs[i])
            if i == len(imgs)-1:
                axes[idx, i].set_title(f"{preprocess_name[1]}+{preprocess_name[2]}\nSize : {imgs[i].size}")
            else:
                axes[idx, i].set_title(f"{preprocess_name[i]}\nSize : {imgs[i].size}")
            axes[idx, i].axis('off')

