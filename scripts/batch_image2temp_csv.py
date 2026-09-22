import os
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

# ================= 1. 路径与参数配置 =================
YOLO_MODEL_PATH = r'/models/best.pt'  # 训练好的YOLO模型路径
PKL_MODEL_PATH = r'/temperature_extraction/getRandomForestRegress/clf_model_RGB_20240906.pkl'  # 温度映射模型路径
IMAGES_DIR = r'/dataset/extracted_frames/20240729T080938-846'  # 需要提取温度图片的文件夹
OUTPUT_CSV = r'E:\real\use_code\yoloV8\dataset\extracted_frames\nostril_temperatures.csv'  # 最终输出的CSV结果文件

RADIUS = 20  # 裁剪鼻孔区域的半径为20像素

# ================= 2. 加载模型 =================
print("正在加载 YOLOv8-pose 模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("正在加载 温度映射模型 (PKL)...")
temp_model = joblib.load(PKL_MODEL_PATH)


# ================= 3. 核心提取函数 =================
def get_circle_temperature(img, center_x, center_y, radius=20, threshold_value=25.953806100238968):
    """
    在图片上以(center_x, center_y)为圆心，抠出半径为radius的圆形区域，
    并把区域内的所有RGB像素喂给PKL模型预测温度，返回平均温度。
    加入阈值过滤：只保留大于 threshold_value 的像素，防止算入冷空气背景！
    """
    h, w = img.shape[:2]

    # 构建一个与原图同样大小的纯黑蒙版（Mask）
    mask = np.zeros((h, w), dtype=np.uint8)

    # 在蒙版上画一个白色的实心圆（代表需要提取的鼻孔区域）
    cv2.circle(mask, (int(center_x), int(center_y)), radius, 255, -1)

    # 提取圆形区域内的所有像素点
    # 注意：这里的像素通道顺序需要和你的 RGB_nihe.py 训练时保持一致
    # 如果训练时直接用的 cv2.imread 没有转格式，这里提取出的就是 B, G, R 特征
    pixels = img[mask == 255]

    if len(pixels) == 0:
        return np.nan

    # 将像素点输入随机森林模型，预测出每一个像素点的温度
    # pixels 的形状是 [N, 3] (N是像素总数，3是通道数)
    temp_predictions = temp_model.predict(pixels)

    # 过滤掉低于 25.95 度的像素（因为牛鼻子内部应该是比较热的，被遮挡或抠到背景时温度会很低）
    filtered_value = temp_predictions[temp_predictions > threshold_value]

    # 如果过滤后一个符合条件的像素都没了（说明完全被遮挡，或者抠到了背景）
    if len(filtered_value) == 0:
        return np.nan

    # 对过滤后的真实热源求平均温度
    avg_temp = round(np.mean(filtered_value), 2)
    return avg_temp


# ================= 4. 遍历图片进行处理 =================
results_data = []

# 获取文件夹里所有的图片并按文件名排序（确保时间顺序正确）
img_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.JPG'))])

print(f"共找到 {len(img_files)} 张图像，开始提取温度...")

for img_name in img_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # 步骤A：用 YOLO 推理，找关键点
    results = yolo_model(img, verbose=False)

    left_temp = np.nan
    right_temp = np.nan

    # 检查是否检测到了目标和关键点
    if len(results) > 0 and len(results[0].keypoints) > 0:
        # 获取包含 (x, y, conf) 的完整数据
        # 获取第一只牛的第一个结果的关键点坐标
        # kpts_data 形状为 [2, 3]，包含了左右鼻孔的 x, y 和 置信度 conf
        kpts_data = results[0].keypoints.data[0].cpu().numpy()

        if len(kpts_data) >= 2:
            # 动态判断模型吐出了几个值
            # 如果模型输出是 [x, y, conf] 3个值
            if len(kpts_data[0]) == 3:
                left_x, left_y, left_conf = kpts_data[0]
                right_x, right_y, right_conf = kpts_data[1]
                # 纯物理级别的遮挡拦截
                if left_conf < 0.5:
                    left_temp = None  # 被遮挡了，不提取

                if right_conf < 0.5:
                    right_temp = None  # 被遮挡了，不提取

            # 如果模型输出只有 [x, y] 2个值（你当前遇到的情况）
            else:
                left_x, left_y = kpts_data[0][:2]
                right_x, right_y = kpts_data[1][:2]
                left_conf, right_conf = 1.0, 1.0  # 伪造一个置信度，防止后续报错

            # 因为我们在 get_circle_temperature 里已经加了 25.95 度的温度过滤机制，
            # 即使被遮挡时模型瞎猜了坐标，抠到了冷空气背景，温度也会被过滤掉并返回 NaN！
            # 所以这里直接大胆计算即可。
            left_temp = get_circle_temperature(img, left_x, left_y, radius=RADIUS)
            right_temp = get_circle_temperature(img, right_x, right_y, radius=RADIUS)

    # 将结果保存
    results_data.append([img_name, left_temp, right_temp])
    print(f"处理完成: {img_name} | 左鼻孔: {left_temp}℃ | 右鼻孔: {right_temp}℃")

# ================= 5. 保存至 CSV =================
df = pd.DataFrame(results_data, columns=['Frame_Name', 'Left_Nostril_Temp', 'Right_Nostril_Temp'])
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n大功告成！所有温度数据已保存至 {OUTPUT_CSV}")