import pandas as pd
import os
import shutil

train_df = pd.read_csv("/data/Computer_Vision/train_df.csv")
print(len(train_df["label"].unique()))  # type = numpy.array
label_list = train_df["label"].unique().tolist()
len(label_list)  # type = list

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
for i in range(len(label_list)):  # 레이블 개수 만큼 
    createFolder(f'/data/Computer_Vision/train/{label_list[i]}')  # 레이블 폴더를 생성 

train_folder = os.listdir('/data/Computer_Vision/train/')
len(train_folder)  # -폴더개수 88 , 사진개수 = 4277개


for i in range(len(train_folder) - 88):  # 폴더 생성한것 88개 뺴주는 겁니다.

    if train_folder[i] == ".png":   # 확장자가 png면 
        label = train_df.loc[train_df["file_name"] == f"{train_folder[i]}"]["label"][i]  # train_df에서 이미지 이름에 맞는 label을 불러와 저장
        file_source = f'/data/Computer_Vision/train/{train_folder[i]}'  # train 폴더에 있는 해당 이미지를
        file_destination = f'/data/Computer_Vision/train/train/{label}/'  # 해당 label 폴더로 이동 
        shutil.move(file_source, file_destination)  # 이동 실행