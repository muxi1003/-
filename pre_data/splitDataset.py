# -*- coding:utf-8 -*-
# @Time:2023/12/11 10:37
# @author:xinchao
# @File:splitDataset.py
# @Software:PyCharm
import os
import shutil
import random
from tqdm import tqdm

dataset_root = r"D:\code\python\project\YOLOV8-Pose-flank\data2YOLO"
# print(os.chdir(os.path.join(dataset_root, 'img')))
# print(os.listdir(os.path.join(dataset_root, 'img')))

# os.chdir(os.path.join(dataset_root, 'images'))
# print(os.listdir(os.path.join(dataset_root, 'images')))


test_frac = 0.2  # 测试集比例
random.seed(123)  # 随机数种子，便于复现

# folder = os.path.join(dataset_root, 'f')
folder = os.path.join(dataset_root, 'images1')
# folder_json = os.path.join(dataset_root, 'labelme_jsons')
img_paths = os.listdir(folder)
random.shuffle(img_paths)  # 随机打乱

val_number = int(len(img_paths) * test_frac)  # 测试集文件个数
train_files = img_paths[val_number:]  # 训练集文件名列表
val_files = img_paths[:val_number]  # 测试集文件名列表

print('数据集文件总数', len(img_paths))
print('训练集文件个数', len(train_files))
print('测试集文件个数', len(val_files))

os.makedirs(os.path.join(folder, "train"), exist_ok=True)
hh = os.path.join(folder, 'train')
os.makedirs(os.path.join(folder, "val"), exist_ok=True)
kk = os.path.join(folder, 'val')

for each in tqdm(train_files):
    gg = os.path.join(folder, each)
    shutil.move(gg, hh)

for each in tqdm(val_files):
    ll = os.path.join(folder, each)
    shutil.move(ll, kk)
#
# os.makedirs(os.path.join(folder_json, "train"), exist_ok=True)
# os.makedirs(os.path.join(folder_json, "val"), exist_ok=True)
#
# hh = os.path.join(folder_json, 'train')
# kk = os.path.join(folder_json, 'val')
# for each in tqdm(train_files):
#     # srt_path = each.split('.')[0] + '.json'
#     gg = os.path.join(folder_json, each)
#     shutil.move(gg, hh)

# for each in tqdm(val_files):
#     # srt_path = each.split('.')[0] + '.json'
#     ll = os.path.join(folder_json, each)
#     shutil.move(ll, kk)
