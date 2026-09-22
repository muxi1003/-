# -*- coding:utf-8 -*-
"""修复标签转换 bug 并重新分配到 train/val。

问题：al_labelme 的矩形是 4-point 格式，旧脚本只取前两个点导致 h=0。
步骤：
  1. 用修复后的逻辑重新转换 al_labelme → al_labelyolo
  2. 根据 images/train 和 images/val 已有的图片，把标签复制到对应的 labels/ 目录
"""
import os
import json
import shutil
from tqdm import tqdm

dataset_root = r"E:\real\use_code\yoloV8\Dataset_new"
labelme_root = os.path.join(dataset_root, "al_labelme")
yolo_label_root = os.path.join(dataset_root, "al_labelyolo")
images_train_dir = os.path.join(dataset_root, "images", "train")
images_val_dir = os.path.join(dataset_root, "images", "val")
labels_train_dir = os.path.join(dataset_root, "labels", "train")
labels_val_dir = os.path.join(dataset_root, "labels", "val")

bbox_class = {"nose": 0}
keypoint_class = ["left_nostril", "right_nostril"]


def process_single_json(labelme_path, save_dir):
    """修复后的 LabelMe → YOLO 转换（兼容 2-point / 4-point 矩形）"""
    with open(labelme_path, "r", encoding="utf-8") as f:
        labelme = json.load(f)

    img_width = labelme["imageWidth"]
    img_height = labelme["imageHeight"]

    yolo_filename = os.path.splitext(os.path.basename(labelme_path))[0] + ".txt"
    yolo_txt_path = os.path.join(save_dir, yolo_filename)
    os.makedirs(save_dir, exist_ok=True)

    with open(yolo_txt_path, "w", encoding="utf-8") as f:
        for each_ann in labelme["shapes"]:
            if each_ann["shape_type"] != "rectangle":
                continue

            # 取所有点的 min/max —— 兼容 2-point 对角 & 4-point 四角
            all_x = [p[0] for p in each_ann["points"]]
            all_y = [p[1] for p in each_ann["points"]]
            bbox_left = int(min(all_x))
            bbox_right = int(max(all_x))
            bbox_top = int(min(all_y))
            bbox_bottom = int(max(all_y))

            bbox_cx = int((bbox_left + bbox_right) / 2)
            bbox_cy = int((bbox_top + bbox_bottom) / 2)
            bbox_w = bbox_right - bbox_left
            bbox_h = bbox_bottom - bbox_top

            yolo_str = "{} {:.5f} {:.5f} {:.5f} {:.5f} ".format(
                bbox_class[each_ann["label"]],
                bbox_cx / img_width,
                bbox_cy / img_height,
                bbox_w / img_width,
                bbox_h / img_height,
            )

            # 找框内的关键点
            kp_dict = {}
            for ann in labelme["shapes"]:
                if ann["shape_type"] == "point":
                    x, y = int(ann["points"][0][0]), int(ann["points"][0][1])
                    if bbox_left < x < bbox_right and bbox_top < y < bbox_bottom:
                        kp_dict[ann["label"]] = [x, y]

            for cls_name in keypoint_class:
                if cls_name in kp_dict:
                    kx, ky = kp_dict[cls_name]
                    yolo_str += "{:.5f} {:.5f} 2 ".format(kx / img_width, ky / img_height)
                else:
                    yolo_str += "0 0 0 "
            f.write(yolo_str + "\n")

    return yolo_txt_path


def copy_labels_for_split(split_img_dir, split_lab_dir):
    """根据已有图片，从 al_labelyolo 子目录找到对应标签并复制"""
    os.makedirs(split_lab_dir, exist_ok=True)
    copied = 0
    for img_name in tqdm(os.listdir(split_img_dir), desc=f"Copy labels -> {os.path.basename(split_lab_dir)}"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        lab_name = os.path.splitext(img_name)[0] + ".txt"
        # 图片文件名格式：{video_id}_frame_xxxxxx.jpg，子文件夹名即 video_id
        video_id = img_name.rsplit("_frame_", 1)[0]
        src_lab = os.path.join(yolo_label_root, video_id, lab_name)
        if os.path.exists(src_lab):
            shutil.copy2(src_lab, os.path.join(split_lab_dir, lab_name))
            copied += 1
        else:
            # 子文件夹名可能是前缀变体（如 bs455, ba455），尝试搜索
            found = False
            for subdir in os.listdir(yolo_label_root):
                candidate = os.path.join(yolo_label_root, subdir, lab_name)
                if os.path.exists(candidate):
                    shutil.copy2(candidate, os.path.join(split_lab_dir, lab_name))
                    copied += 1
                    found = True
                    break
            if not found:
                print(f"  未找到标签: {lab_name} (video={video_id})")
    print(f"  共复制 {copied} 个标签")


# ===================== 步骤 1：重新转换所有 LabelMe → YOLO =====================
print("=" * 50)
print("步骤 1：重新转换 LabelMe JSON → YOLO txt")
print("=" * 50)

subdirs = sorted(os.listdir(labelme_root))
for subdir in subdirs:
    src_dir = os.path.join(labelme_root, subdir)
    if not os.path.isdir(src_dir):
        continue
    json_files = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    if not json_files:
        continue
    save_dir = os.path.join(yolo_label_root, subdir)
    print(f"\n{subdir} ({len(json_files)} files)")
    for jf in tqdm(json_files, desc=f"  Converting"):
        process_single_json(os.path.join(src_dir, jf), save_dir)
print("\n转换完成！")

# ===================== 步骤 2：复制标签到 train/val =====================
print("\n" + "=" * 50)
print("步骤 2：复制标签到 labels/train 和 labels/val")
print("=" * 50)

copy_labels_for_split(images_train_dir, labels_train_dir)
copy_labels_for_split(images_val_dir, labels_val_dir)

print("\n全部完成！")
