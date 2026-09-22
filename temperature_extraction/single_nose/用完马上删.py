from PIL import Image
import csv

# 打开 JPG 图像
image_path = r'H:/实验数据整理/林甸县奶牛面部热红外/利用关键点进行奶牛鼻孔定位的YOLOv8相关结果/0130_yolov8/yoloV8/temperature_extraction/getRandomForestRegress/20230810T152315.JPG'  # 替换为你的图像文件路径
img = Image.open(image_path)

# 确保图像尺寸是 640x480
img = img.resize((640, 480))

# 获取图像的像素数据
pixels = img.load()

# 创建 CSV 文件以写入像素的 RGB 值
csv_file_path = 'pixels_rgb.csv'  # 输出 CSV 文件路径
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X', 'Y', 'R', 'G', 'B'])  # 写入表头

    # 遍历图像的每个像素
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pixels[x, y]  # 获取 RGB 值
            writer.writerow([x, y, r, g, b])  # 写入 CSV 文件

print(f"RGB values saved to {csv_file_path}")