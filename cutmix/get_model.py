import numpy as np
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torch.nn import functional as F
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.model_selection import StratifiedKFold

from cutmix import rand_bbox
from data_load import get_data_loader

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(args):
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, args["FEATURE_EXTRACTING"])
    model.fc = nn.Linear(512, 88)
    model = model.to(args["DEVICE"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args["LEARNING_RATE"], weight_decay=args["WEIGHT_DECAY"])
    
    return model, criterion, optimizer

def check_batch(model, train_loader,criterion, optimizer, args):
    
    image, label = next(iter(train_loader))

    for i in range(30):
        data = image.to(args["DEVICE"])
        targets = label.to(args["DEVICE"])

        outputs = model(data)
        loss = criterion(outputs, torch.max(targets,1)[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    return 

def validation(model, valid_loader, criterion):
    
    accuracy = 0
    valid_loss = 0
    y_score = []

    for i, (X, y) in enumerate(valid_loader):
        if torch.cuda.is_available():
            X = X.to('cuda')
            y = y.to('cuda')

        outputs = model(X)
        loss = criterion(outputs, torch.max(y,1)[1])
        valid_loss += loss.item()
        outputs_ = torch.argmax(outputs, dim=1)
        
        accuracy += (outputs_ == y).float().sum()
        
        # For roc_auc_score
        outputs = F.softmax(outputs, dim=-1)
        y_score.append(outputs.detach().cpu().numpy())
    
    y_score = np.array([i for sub in y_score for i in sub])

    return valid_loss, accuracy, y_score

def train_model(model, train_loader, valid_loader, criterion, optimizer, args, fold_num=1):
    steps = 0
    total_step = len(train_loader)
    train_losses, validation_losses = [], []
    best_val = np.inf
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True)
    
    if torch.cuda.is_available():
        model = model.to(args["DEVICE"])

    for epoch in range(args['NUM_EPOCHS']):
        running_loss = 0
        for i, (X, y) in enumerate(train_loader):
            
            if torch.cuda.is_available():
                X = X.to(args["DEVICE"])
                y = y.to(args["DEVICE"])

                
            if args["BETA"] > 0 and np.random.random()>0.5: # cutmix 작동될 확률      
                lam = np.random.beta(args["BETA"], args["BETA"])
                rand_index = torch.randperm(X.size()[0]).to(args["DEVICE"])
                target_a = y
                target_b = y[rand_index]            
                bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
                outputs = model(X)
                loss = criterion(outputs, torch.max(target_a,1)[1]) * lam + criterion(outputs, torch.max(target_b,1)[1]) * (1. - lam)
                
            else:
                outputs = model(X)
                loss = criterion(outputs, torch.max(y,1)[1])                  
                
            steps += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % total_step == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy, y_score = validation(model, valid_loader, criterion)

                print("Epoch: {}/{}.. ".format(epoch + 1, args['NUM_EPOCHS']) +
                      "Training Loss: {:.5f}.. ".format(running_loss / total_step) +
                      "Valid Loss: {:.5f}.. ".format(valid_loss / len(valid_loader)) +
                      "Valid Accuracy: {:.5f}.. ".format(accuracy / len(valid_loader.dataset)) ) # 전체 데이터 수에 대해 나눠준다
                
                
                # Save Model
                if (valid_loss / len(valid_loader)) < best_val:
                    best_val = (valid_loss / len(valid_loader))
                    torch.save(model.state_dict(), f"{args['MODEL_PATH']}/{fold_num}_best_checkpoint_{str(epoch + 1).zfill(3)}epoch.tar")
                    try:
                        os.remove(f_pth)
                    except:
                        pass
                    f_pth = f"{args['MODEL_PATH']}/{fold_num}_best_checkpoint_{str(epoch + 1).zfill(3)}epoch.tar"
                
                train_losses.append(running_loss / len(train_loader))
                validation_losses.append(valid_loss / len(valid_loader))
                steps = 0
                running_loss = 0
                model.train()
                
        scheduler.step(valid_loss / len(valid_loader))

    return 

def fold_train(args, train_df):
    folds = StratifiedKFold(n_splits=args["NUM_FOLDS"], shuffle=True, random_state=42)

    X = train_df

    for i, (train_index, valid_index) in enumerate(folds.split(X,X["label_num"])):
        fold_num = i+1
        X_train = X.iloc[train_index]
        X_val = X.iloc[valid_index]


        model, criterion, optimizer = get_model(args)
        train_loader, valid_loader = get_data_loader(X_train, X_val)

        print("=" * 100)
        print(f"{fold_num}/{args['NUM_FOLDS']} Cross Validation Training Starts ...\n")
        train_model(model, train_loader, valid_loader, criterion, optimizer, args, fold_num=fold_num)
        print(f"\n{fold_num}/{args['NUM_FOLDS']} Cross Validation Training Ends ...\n")

    return