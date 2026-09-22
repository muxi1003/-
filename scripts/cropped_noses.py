import os
import cv2
import numpy as np
from ultralytics import YOLO

# ================= 1. 路径配置 =================
YOLO_MODEL_PATH = r'/models/best.pt'  # 你的YOLO模型
IMAGES_DIR = r'/dataset/extracted_frames/20240729T080938-846'
OUTPUT_DIR = r'/dataset/extracted_frames/cropped_noses'  # 抠图输出的文件夹

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

yolo_model = YOLO(YOLO_MODEL_PATH)


# ================= 2. 抠图函数 =================
def crop_circle(img, cx, cy, radius=20):
    """
    以 (cx, cy) 为圆心抠出半径 20 的圆。
    返回一张 40x40 的小图，圆内是原图，圆外是纯黑 [0,0,0]。
    """
    h, w = img.shape[:2]
    cx, cy = int(cx), int(cy)

    # 创建一个 40x40 的纯黑底板
    patch_size = radius * 2
    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

    # 计算原图上的边界（防止圆心在图片边缘导致越界报错）
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)

    # 计算在 40x40 底板上的相对边界
    px1, py1 = radius - (cx - x1), radius - (cy - y1)
    px2, py2 = radius + (x2 - cx), radius + (y2 - cy)

    # 将原图的矩形区域贴到底板上
    patch[py1:py2, px1:px2] = img[y1:y2, x1:x2]

    # 制作一个圆形的掩膜（Mask）
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    # 应用掩膜，把圆外面的矩形四个角变成纯黑色 [0,0,0]
    final_patch = cv2.bitwise_and(patch, patch, mask=mask)
    return final_patch


# ================= 3. 遍历图片并抠图 =================
img_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.JPG'))])
print(f"开始处理 {len(img_files)} 张图片...")

for img_name in img_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None: continue

    results = yolo_model(img, verbose=False)

    if len(results) > 0 and len(results[0].keypoints) > 0:
        kpts_data = results[0].keypoints.data[0].cpu().numpy()

        if len(kpts_data) >= 2:
            # 获取坐标
            left_x, left_y = kpts_data[0][:2]
            right_x, right_y = kpts_data[1][:2]

            # 抠出左鼻孔并保存
            left_patch = crop_circle(img, left_x, left_y, radius=20)
            left_save_path = os.path.join(OUTPUT_DIR, f"left_{img_name}")
            cv2.imwrite(left_save_path, left_patch)

            # 抠出右鼻孔并保存
            right_patch = crop_circle(img, right_x, right_y, radius=20)
            right_save_path = os.path.join(OUTPUT_DIR, f"right_{img_name}")
            cv2.imwrite(right_save_path, right_patch)

print(f"抠图完成！所有小图已保存在：{OUTPUT_DIR} 文件夹下。")