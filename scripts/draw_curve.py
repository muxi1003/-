import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] =['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False

# 读取我们提取好的温度数据
df = pd.read_csv(r'C:\Users\muxi\Desktop\images1.csv')

plt.figure(figsize=(12, 5))
# 将空值(NaN)填为0
plt.plot(df.index, df['Left_Nostril_Temp'].fillna(0), label='左鼻孔温度 (Left Temperature)', marker='.', markersize=6)
plt.plot(df.index, df['Right_Nostril_Temp'].fillna(0), label='右鼻孔温度 (Right Temperature)', marker='.', markersize=6)

plt.title('左右侧鼻孔温度变化原始曲线', fontsize=16)
plt.xlabel('帧数 (Frames)', fontsize=14)
plt.ylabel('温度 (°C)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 保存绘制图片图
plt.savefig(r'C:\Users\muxi\Desktop\images1.png', dpi=600, bbox_inches='tight')
plt.show()