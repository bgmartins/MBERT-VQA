import random
import pandas as pd
import os
import glob
from PIL import Image

data_dir = os.path.join('..', 'datasets/RSIVQA')

dataset1 = open(os.path.join(data_dir, 'AID/rsicd_vqa.txt')).readlines()
dataset2 = open(os.path.join(data_dir, 'DOTA/dota_train_vqa2.txt')).readlines()
dataset3 = open(os.path.join(data_dir, 'DOTA/dota_val_vqa2.txt')).readlines()
dataset4 = open(os.path.join(data_dir, 'HRRSD/opt_val_vqa.txt')).readlines()
dataset5 = open(os.path.join(data_dir, 'Sydney/sydney_vqa.txt')).readlines()
dataset6 = open(os.path.join(data_dir, 'UCM/ucm_res.txt')).readlines()
dataset7 = open(os.path.join(data_dir, 'UCM/ucm_vqa.txt')).readlines()

dataset1[0] = "mode,img_id,category,question,answer\n"
dataset2[0] = "mode,img_id,category,question,answer\n"
dataset3[0] = "mode,img_id,category,question,answer\n"
dataset4[0] = "mode,img_id,category,question,answer\n"
dataset5[0] = "mode,img_id,category,question,answer\n"
dataset6[0] = "mode,img_id,category,question,answer\n"
dataset7[0] = "mode,img_id,category,question,answer\n"

for pos,str in enumerate(dataset1):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset1[pos] = "test,aid_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]
for pos,str in enumerate(dataset2):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset2[pos] = "test,dota_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]
for pos,str in enumerate(dataset3):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset3[pos] = "test,dota_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]
for pos,str in enumerate(dataset4):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset4[pos] = "test,opt_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]
for pos,str in enumerate(dataset5):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset5[pos] = "test,sydney_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]
for pos,str in enumerate(dataset6):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset6[pos] = "test,ucm_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]
for pos,str in enumerate(dataset7):
    if pos==0: continue
    if str.endswith("?yes\n") or str.endswith("?no\n"): category = "yes_no"
    else: category = "not_yes_no"
    dataset7[pos] = "test,ucm_" + str[:str.index(":")] + "," + category + "," + str[str.index(":")+1:str.index("?")+1] + "," + str[str.index("?")+1:]

aux_data = dataset1[1:] + dataset2[1:] + dataset3[1:] + dataset4[1:] + dataset5[1:] + dataset6[1:] + dataset7[1:]
random.shuffle(aux_data)
size1 = int(len(aux_data) * 0.8)
size2 = int(len(aux_data) * 0.1)
train_data = [ dataset1[0] ] + aux_data[0:size1]
test_data = [ dataset1[0] ] + aux_data[size1+1:size1+size2]
val_data = [ dataset1[0] ] + aux_data[size1+size2+1:]

open(os.path.join(data_dir, 'traindf.csv'), 'w').writelines(train_data)
open(os.path.join(data_dir, 'valdf.csv'), 'w').writelines(val_data)
open(os.path.join(data_dir, 'testdf.csv'), 'w').writelines(test_data)
