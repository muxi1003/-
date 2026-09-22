"""
Created on 0906,用于验证热红外图像随机模型的温度映射准确度；
"""
import os
import re
import cv2
import time
import joblib
import datetime
import numpy as np
import scipy.signal
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


def read_csv_my(csv_path):
    data = pd.read_csv(csv_path, header=None)
    data = data.to_numpy().reshape(-1, 1)
    return data


def calcTemp(img_path, clf_model):
    print(f"正在处理: {os.path.basename(img_path)}")
    img1 = cv2.imread(img_path)
    if img1 is None:
        print(f"警告: 无法读取图片 {img_path}")
        return None

    print(f"图片尺寸: {img1.shape}, 总像素数: {img1.shape[0] * img1.shape[1]}")

    # 使用向量化操作提高性能
    # 将图片重塑为 (像素数, 3) 的形状
    pixels = img1.reshape(-1, 3)

    # 批量预测所有像素
    print("开始批量预测...")
    start_time = time.time()
    predictions = clf_model.predict(pixels)
    elapsed_time = time.time() - start_time
    print(f"预测完成，耗时: {elapsed_time:.2f}秒")

    return predictions.reshape(-1, 1)

# def calcTemp(img_path):
#     clf_model = joblib.load('C:/cxc/code/python/project_all/yoloV8/temperature_extraction/getRandomForestRegress/clf_model_RGB_1214.pkl')
#
#     img1 = cv2.imread(img_path)
#     height, width, _ = img1.shape
#     temperatures = []  # 存储有效像素点的温度值
#     for y in range(height):
#         for x in range(width):
#             pixel = img1[y, x]
#             if pixel[0] == 0:
#                 continue  # 如果任一通道的值为0，则跳过当前像素点的温度计算
#             prediction = clf_model.predict(pixel.reshape(1, -1))
#             temperatures.append(prediction)
#
#     if len(temperatures) > 0:
#         temperatures = np.array(temperatures).reshape(-1, 1)
#         return temperatures
#     else:
#         return None  # 如果没有有效像素点，则返回None表示跳过计算


# 读取文件夹中所有图片文件
def read_and_sort_images(folder_path, clf_model, max_images=None):
    # 读取文件夹下的所有文件名
    files = os.listdir(folder_path)
    # 过滤出图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
    image_files = [f for f in files if os.path.splitext(f)[1] in image_extensions]

    # 如果指定了最大图片数，则只处理前N张
    if max_images is not None:
        image_files = image_files[:max_images]

    all_temperatures = []
    for img in tqdm(image_files, desc="处理图片"):
        img_path = os.path.join(folder_path, img)
        temperature = calcTemp(img_path, clf_model)
        if temperature is not None:
            all_temperatures.extend(temperature.flatten().tolist())

    return np.array(all_temperatures).reshape(-1, 1) if all_temperatures else None


if __name__ == '__main__':
    # 加载模型
    clf_model = joblib.load(r'E:\real\use_code\yoloV8\temperature_extraction\getRandomForestRegress\clf_model_RGB_20240906.pkl')

    folder_path = r'E:\real\use_code\yoloV8\temperature_extraction\getRandomForestRegress'
    # 只处理第一张图片进行测试
    data_test = read_and_sort_images(folder_path, clf_model, max_images=1)

    csv_path = r'E:\real\use_code\yoloV8\temperature_extraction\getRandomForestRegress\1.csv'
    data_real = read_csv_my(csv_path)

    print(f"data_test为{type(data_test)}, 形状为{data_test.shape if data_test is not None else 'None'}")
    print(f"data_real为{type(data_real)}, 形状为{data_real.shape}")

    if data_test is not None and len(data_test) == len(data_real):
        # 计算R²分数
        r2 = r2_score(data_real, data_test)
        print("R2 Score:", r2)

        # 计算MAE
        mae = mean_absolute_error(data_real, data_test)
        print("MAE:", mae)
    else:
        print("数据长度不匹配或数据为空，无法计算指标")

# 0.83359  0.923243
# 0.88283  0.87703
#

