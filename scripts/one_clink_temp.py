import os
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

# ================= 1. 路径配置 =================
YOLO_MODEL_PATH = r'E:\real\use_code\yoloV8\models\best.pt'
PKL_MODEL_PATH = r'E:\real\use_code\yoloV8\temperature_extraction\getRandomForestRegress\clf_model_RGB_20240906.pkl'
IMAGES_DIR = r"E:\real\use_code\yoloV8\Dataset_new\72video\al_images\zs16110577"  # 你的视频原提取帧图文件夹

# ================= 2. 双重保险参数 (核心！) =================
CONF_THRESH = 0.8  # 第一重锁：YOLO置信度必须大于 0.8 (真鼻子一般在0.9以上，假鼻子通常在0.5~0.7)
TEMP_THRESH = 30.5  # 第二重锁：像素温度必须大于 30.5 度 (牛脸皮毛一般低于此温度)
RADIUS = 20  # 抠图半径

print("正在加载 YOLOv8 和 PKL 模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)
temp_model = joblib.load(PKL_MODEL_PATH)


def get_real_temperature(img, cx, cy):
    """在原图上抠圆并计算温度，加入温度拦截"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), RADIUS, 255, -1)

    # 布尔索引 3 通道图直接返回 (N, 3)
    pixels = img[mask == 255]
    if len(pixels) == 0:
        return None

    predictions = temp_model.predict(pixels)

    # 第二重锁：只保留大于 30.5 度的有效鼻孔高温像素
    valid_temps = predictions[predictions > TEMP_THRESH]

    if len(valid_temps) == 0:
        return None  # 即使YOLO瞎猜了，只要温度不够，依然拦截！

    return round(np.mean(valid_temps), 2)


# ================= 3. 开始处理 =================
img_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
results_data = []

print(f"开始提取，共检测到 {len(img_files)} 帧画面...")

for img_name in img_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[警告] 读取失败: {img_name}")
        continue
    # cv2 读入为 BGR，转 RGB 供模型使用（若模型训练时已是 BGR 则注释此行）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = yolo_model(img, verbose=False)
    left_temp, right_temp = None, None

    if len(results) > 0 and len(results[0].keypoints) > 0:
        kpts_data = results[0].keypoints.data[0].cpu().numpy()

        if len(kpts_data) >= 2:
            left_x, left_y, left_conf = kpts_data[0]
            right_x, right_y, right_conf = kpts_data[1]

            # 第一重锁：置信度必须大于 0.8
            if left_conf > CONF_THRESH:
                left_temp = get_real_temperature(img, left_x, left_y)

            if right_conf > CONF_THRESH:
                right_temp = get_real_temperature(img, right_x, right_y)

    print(f"{img_name} | 左: {left_temp} | 右: {right_temp}")
    results_data.append([img_name, left_temp, right_temp])

# ================= 4. 保存结果 =================
csv_save_path = r'E:\real\use_code\yoloV8\Dataset_new\72video\al_images\zs16110577\images1.csv'
df = pd.DataFrame(results_data, columns=['Frame_Name', 'Left_Nostril_Temp', 'Right_Nostril_Temp'])
df.to_csv(csv_save_path, index=False)
print(f"\n大功告成！完美数据已保存至: {csv_save_path}")