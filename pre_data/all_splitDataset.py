# -*- coding:utf-8 -*-
import os
import shutil
import random
import yaml
from tqdm import tqdm


dataset_root = r"E:\real\use_code\yoloV8\Dataset_new"  # 数据集根目录
images_dir = os.path.join(dataset_root, "images1")  # 图片文件夹
labels_dir = os.path.join(dataset_root, "labels1")  # 标签文件夹
test_frac = 0.2     # 验证集比例
random.seed(123)

# 关键点类别
class_names = ["nose"]  # 检测框：鼻子框
yaml_save_path = os.path.join(dataset_root, "dataset.yaml")

# 1. 获取所有图片文件名
img_files = [f for f in os.listdir(images_dir) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]
random.shuffle(img_files)

# 2. 划分训练集/验证集
val_count = int(len(img_files) * test_frac)
train_img_files = img_files[val_count:]
val_img_files = img_files[:val_count]

print("总图片数：", len(img_files))
print("训练集图片数：", len(train_img_files))
print("验证集图片数：", len(val_img_files))

# 3. 创建 YOLO 标准目录结构
train_img_dir = os.path.join(dataset_root, "images", "train")
val_img_dir = os.path.join(dataset_root, "images", "val")
train_lab_dir = os.path.join(dataset_root, "labels", "train")
val_lab_dir = os.path.join(dataset_root, "labels", "val")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lab_dir, exist_ok=True)
os.makedirs(val_lab_dir, exist_ok=True)

# ------------------- 移动 训练集 图片 + 标签 -------------------
print("\n正在移动训练集...")
for img_name in tqdm(train_img_files):
    # 移动图片
    src_img = os.path.join(images_dir, img_name)
    dst_img = os.path.join(train_img_dir, img_name)
    shutil.move(src_img, dst_img)

    # 移动对应的标签
    lab_name = os.path.splitext(img_name)[0] + ".txt"
    src_lab = os.path.join(labels_dir, lab_name)
    dst_lab = os.path.join(train_lab_dir, lab_name)
    if os.path.exists(src_lab):
        shutil.move(src_lab, dst_lab)

# ------------------- 移动 验证集 图片 + 标签 -------------------
print("\n正在移动验证集...")
for img_name in tqdm(val_img_files):
    # 移动图片
    src_img = os.path.join(images_dir, img_name)
    dst_img = os.path.join(val_img_dir, img_name)
    shutil.move(src_img, dst_img)

    # 移动对应的标签
    lab_name = os.path.splitext(img_name)[0] + ".txt"
    src_lab = os.path.join(labels_dir, lab_name)
    dst_lab = os.path.join(val_lab_dir, lab_name)
    if os.path.exists(src_lab):
        shutil.move(src_lab, dst_lab)

# ------------------- 生成 YOLOv8 Pose 所需 yaml -------------------
yaml_content = {
    "path": dataset_root,
    "train": "images/train",
    "val": "images/val",
    "nc": 1,                     # 检测框类别：nose
    "names": ["nose"],           # 类别名
    "kpt_shape": [2, 3],         # 2个关键点，每个 (x,y,Visibility)
    "flip_idx": [1, 0]           # 左右鼻孔翻转对应（必须加！）
}

with open(yaml_save_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_content, f, indent=2, sort_keys=False, allow_unicode=True)

print("\n数据集切分完成！")
print("标签同步切分完成！")
print("dataset.yaml 已生成：", yaml_save_path)