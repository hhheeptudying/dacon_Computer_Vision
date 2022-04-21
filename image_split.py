
import pandas as pd
import os
import shutil

data_path = '/home/user/PycharmProjects/pythonProject/dacon/dacon_Computer_Vision'
train_df = pd.read_csv(data_path + "/open/train_df.csv")
print(len(train_df["label"].unique())) 
label_list = train_df["label"].unique().tolist()
len(label_list)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)
 
for i in range(len(label_list)):
    createFolder(data_path + f'/open/split/{label_list[i]}')  

train_folder = os.listdir(data_path + '/open/train/')
print(len(train_folder)) 

for i in range(len(train_folder)):
    if train_folder[i][-3:] == "png": 
        label = train_df.loc[train_df["file_name"] == train_folder[i]]["label"].values[0]
        file_source = data_path + '/open/train/'+ train_folder[i] 
        file_destination = data_path + '/open/split/'+label  
        shutil.copy(file_source, file_destination) 