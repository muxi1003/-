import os
import cv2
import numpy as np
import pandas as pd
import joblib

# ================= 1. 路径配置 =================
# 模型路径
PKL_MODEL_PATH = r"/temperature_extraction/getRandomForestRegress/clf_model_RGB_20240906.pkl"
# 刚才抠出来的圆鼻孔文件夹路径
CROPPED_DIR = r"/dataset/extracted_frames/cropped_noses"

print("正在加载 温度映射模型 (PKL)...")
temp_model = joblib.load(PKL_MODEL_PATH)


# ================= 2. 核心温度计算函数 =================
def my_calcTemp(img_path, threshold_value=30):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 智能兼容：我们不知道原作者的 pkl 是用 3通道(RGB) 还是 1通道 训练的，所以用 try-except 保护
    try:
        # 尝试按照 3通道(BGR) 输入给模型
        pixels = img.reshape(-1, 3)
        predictions = temp_model.predict(pixels)
    except ValueError:
        # 如果报错，说明原作者是用 1通道（灰度/展平）训练的，兼容原作者的写法
        pixels = img.reshape(-1, 1)
        predictions = temp_model.predict(pixels)

    # 核心过滤逻辑：过滤掉图片周围的纯黑背景，以及低于 30度 的冷空气/牛毛
    valid_temps = predictions[predictions > threshold_value]

    # 如果过滤后什么都不剩（说明鼻孔被完全遮挡，只拍到了背景或牛毛）
    if len(valid_temps) == 0:
        return None

    # 计算并返回平均温度
    return round(np.mean(valid_temps), 2)


# ================= 3. 遍历图片并输出 CSV =================
img_files = os.listdir(CROPPED_DIR)
# 提取出每一帧的原始名字
original_names = sorted(list(set([
    f.replace('left_', '').replace('right_', '')
    for f in img_files if f.endswith(('.jpg', '.png', '.JPG'))
])))

results_data = []
print(f"\n开始提取温度，共检测到 {len(original_names)} 帧画面...")

for name in original_names:
    left_path = os.path.join(CROPPED_DIR, f"left_{name}")
    right_path = os.path.join(CROPPED_DIR, f"right_{name}")

    left_temp = None
    right_temp = None

    # 计算左鼻孔
    if os.path.exists(left_path):
        left_temp = my_calcTemp(left_path, threshold_value=30)

    # 计算右鼻孔
    if os.path.exists(right_path):
        right_temp = my_calcTemp(right_path, threshold_value=30)

    print(f"画面: {name} | 左鼻孔: {left_temp} | 右鼻孔: {right_temp}")
    results_data.append([name, left_temp, right_temp])

# ================= 4. 保存最终结果 =================
csv_save_path = os.path.join(os.path.dirname(CROPPED_DIR), 'final_temperatures.csv')
df = pd.DataFrame(results_data, columns=['Frame_Name', 'Left_Nostril_Temp', 'Right_Nostril_Temp'])
df.to_csv(csv_save_path, index=False)

print(f"\n太棒了！全部温度提取完成！")
print(f"数据已保存至: {csv_save_path}")