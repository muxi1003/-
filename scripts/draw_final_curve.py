import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ================= 1. 图表显示中文字体配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 读取我们提取好的CSV =================
print("正在读取温度数据...")
df = pd.read_csv(r'E:\real\use_code\yoloV8\Dataset_new\72video\al_images\zs16110577\images1.csv')

# 提取左右鼻孔温度（此时遮挡的还是 NaN 空值）
left_temp = df['Left_Nostril_Temp'].values
right_temp = df['Right_Nostril_Temp'].values

# ================= 3. 真实的温度融合与遮挡插值 =================
total_raw_temp =[]
for l, r in zip(left_temp, right_temp):
    if pd.notna(l) and pd.notna(r):
        total_raw_temp.append(max(l, r)) # 都在时取最高温
    elif pd.notna(l):
        total_raw_temp.append(l)
    elif pd.notna(r):
        total_raw_temp.append(r)
    else:
        total_raw_temp.append(np.nan)    # 完全遮挡时，保持为空

# 使用 Pandas 强大的线性插值，把遮挡时的断点平滑地连接起来
temp_series = pd.Series(total_raw_temp)
temp_series = temp_series.interpolate(method='linear').bfill().ffill()
total_temp_filled = temp_series.values

# ================= 4. 统一归一化 =================
min_val = np.min(total_temp_filled)
max_val = np.max(total_temp_filled)
if max_val > min_val:
    norm_temp = (total_temp_filled - min_val) / (max_val - min_val)
else:
    norm_temp = total_temp_filled

# ================= 5. 加大平滑力度 =================
# 将窗口改回 5，既能滤除相机噪点，又不会把真实的呼吸波浪给抹平
window_size = 5
smoothed_temp = np.convolve(norm_temp, np.ones(window_size) / window_size, mode='valid')

# ================= 6. 【核心优化3】严格的寻峰门槛 =================
# distance=8 : 两次呼吸波峰至少间隔 8 帧 (约 0.93秒，完美符合成年牛最快呼吸频率极限)
# prominence=0.03 : 降低突出度要求，精准抓取 400~500 帧那里的真实呼吸起伏！
peaks, _ = find_peaks(smoothed_temp, distance=8, prominence=0.03)

# ================= 7. 计算精准呼吸率 (RR) =================
extracted_fps = 8.57  # 使用你实际提取的真实帧率
total_frames = len(smoothed_temp)
video_duration_minutes = (total_frames / extracted_fps) / 60
respiratory_rate = len(peaks) / video_duration_minutes if video_duration_minutes > 0 else 0

print(f"\n数据处理完毕！")
print(f"视频总帧数(平滑后): {total_frames} 帧")
print(f"检测到真实呼吸次数: {len(peaks)} 次")
print(f"最终计算的呼吸率 (RR): {respiratory_rate:.2f} 次/分钟")

# ================= 8. 绘制图表 =================
plt.figure(figsize=(14, 6))
x_axis = np.arange(len(smoothed_temp))

# 画平滑后的主曲线
plt.plot(x_axis, smoothed_temp, color='#1f77b4', linewidth=2.5, label='融合平滑后温度曲线')

# 在波峰处打上红色的叉叉
plt.plot(peaks, smoothed_temp[peaks], "x", color='red', markersize=12, markeredgewidth=2.5, label='检测到的呼吸波峰')

plt.title(f'奶牛鼻孔温度波动曲线 (推算呼吸率: {respiratory_rate:.1f} 次/分钟)', fontsize=20, pad=15)
plt.xlabel('帧数 (Frames)', fontsize=16)
plt.ylabel('归一化温度幅值', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存高标清图片
output_pic = r'E:\real\use_code\yoloV8\Dataset_new\72video\al_images\zs16110577\Final_Respiration_Curve_Pro.png'
plt.savefig(output_pic, dpi=600, bbox_inches='tight')
print(f"高清曲线图已成功保存为: {output_pic}")