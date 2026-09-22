import os
import re
import cv2
import sys
import time
import glob
import datetime
import joblib
import openpyxl
import requests
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib
import matplotlib

matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'GTK3Agg', 'macosx' 等，视你的系统而

import numpy as np
import matplotlib.pyplot as plt
# from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks

from matplotlib import pyplot as plt
import concurrent.futures
from optparse import OptionParser
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt


def load_model(func):
    clf_model = joblib.load(
        r"C:\cxc\code\python\project_all\yoloV8\temperature_extraction\getRandomForestRegress\clf_model_RGB_1214.pkl")

    def wrapper(img_path, threshold_value=32):
        img1 = cv2.imread(img_path)
        x_test = img1.reshape(-1, 3)
        predictions = clf_model.predict(x_test)
        filetered_value = predictions[predictions > threshold_value]
        average = np.mean(filetered_value)
        return average

    return wrapper


@load_model
def calcTemp(img_path, clf_model):
    img1 = cv2.imread(img_path)
    height, width, _ = img1.shape
    temperatures = []  # 存储有效像素点的温度值
    for y in range(height):
        for x in range(width):
            pixel = img1[y, x]
            if pixel[0] == 0:
                continue  # 如果任一通道的值为0，则跳过当前像素点的温度计算
            # x_test = pixel.reshape(1, -1)
            prediction = clf_model.predict(pixel)
            # if prediction > threshold_value:
            temperatures.append(prediction)

    if len(temperatures) > 0:
        average = np.mean(temperatures)
        return average
    else:
        return None  # 如果没有有效像素点，则返回None表示跳过计算


# 读取文件夹中所有图片文件
def read_and_sort_images(folder_path):
    # 读取文件夹下的所有文件名
    files = os.listdir(folder_path)

    # 过滤出图片文件并提取图片前缀数字
    images = [(file, int(re.match(r'(\d+)_', file).group(1))) for file in files if file.endswith('.png')]

    # 根据图片前缀数字排序
    images.sort(key=lambda x: x[1])

    # 查找最大的前缀数字
    mmath = max(images, key=lambda x: x[1])[1] if images else 0

    # 根据最大的前缀数字初始化温度列表，填充数字零
    left_temp = [0] * mmath
    right_temp = [0] * mmath

    # 遍历排序后的图片列表
    for image_name, prefix_number in images:
        prefix_number = prefix_number - 1
        print(prefix_number)
        # 计算图片温度
        image_path = os.path.join(folder_path, image_name)
        print(image_path)
        temperature = calcTemp(image_path)
        print(prefix_number)

        # 根据图片名称中的尾标填充对应的温度列表
        if image_name.endswith('left.png'):
            left_temp[prefix_number] = temperature
        elif image_name.endswith('right.png'):
            right_temp[prefix_number] = temperature

    return left_temp, right_temp, mmath


# 定义一个函数来填充0值
def fill_zeros_with_average(temp_list):
    filled_temp = temp_list[:]  # 复制列表，以便我们可以修改值而不改变原始列表
    for i in range(1, len(temp_list) - 1):  # 从第二个元素到倒数第二个元素
        if temp_list[i] == 0 and temp_list[i - 1] != 0 and temp_list[i + 1] != 0:
            filled_temp[i] = (temp_list[i - 1] + temp_list[i + 1]) / 2
    return filled_temp


# 定义移动平均函数，特殊处理边界
def moving_average(data, window_size):
    # 扩展数据：前后各复制window_size//2个数据点
    extension = np.r_[data[window_size - 1:0:-1], data, data[-1:-window_size:-1]]
    # 应用移动平均
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(extension, window, mode='valid')
    return smoothed[window_size // 2:-window_size // 2 + 1]


# 应用移动平均去噪声
def numPeak(datalist: list):
    fenmu = len(datalist)
    data1 = np.array(datalist)
    # peaks, _ = find_peaks(data1, distance=desired_distance)  # , distance=8, height=34.2
    # peaks, _ = find_peaks(data1, height=None, threshold=None, distance=4,
    #            prominence=0.035, width=2, wlen=None,
    #            plateau_size=None)

    peaks, _ = find_peaks(data1, height=None, threshold=None, distance=4,
                          prominence=0.035, width=1, wlen=None,
                          plateau_size=1)
    fenzi = len(peaks)
    fenmu = (fenmu / 8.6) / 60
    respiratory_rate = fenzi / fenmu
    return peaks, respiratory_rate, fenzi


def pinjie_LR(left_zero1, right_zero1):
    left_temp = left_zero1
    right_temp = right_zero1

    # 找到最小和最大温度值，排除为0的值
    try:
        non_zero_min_l = min(filter(lambda x: x != 0, left_temp))
    except:
        non_zero_min_l = left_temp

    try:
        non_zero_min_r = min(filter(lambda x: x != 0, right_temp))
    except:
        non_zero_min_r = right_temp

    max_temp_l = max(left_temp)
    max_temp_r = max(right_temp)


    # 归一化处理
    try:
        left_temp = [(temp - non_zero_min_l) / (max_temp_l - non_zero_min_l) if temp - non_zero_min_l > 0 else 0 for temp in left_temp]
    except:
        print("hhhhh_left")

    print(f"left_temp函数内为{left_temp}")

    try:
        right_temp = [(temp - non_zero_min_r) / (max_temp_r - non_zero_min_r) if temp - non_zero_min_r > 0 else 0 for temp in right_temp]
    except:
        print("hhhhh_right")

    print(f"right_temp函数内为{right_temp}")


    # right_temp = [(temp - non_zero_min_r) / (max_temp_r - non_zero_min_r) for temp in right_temp]
    # print(f"left_temp为  {left_temp}")
    first = True
    isLeft = False
    colors = []
    count = 0

    total_temp = []

    for i in range(len(left_temp)):
        if first:
            if left_temp[i] > 0 and right_temp[i] > 0:
                first = False
                total_temp.append((left_temp[i] + right_temp[i]) / 2)
                # isLeft = True
                colors.append(2)
            elif left_temp[i] > 0 and right_temp[i] == 0:
                first = False
                total_temp.append(left_temp[i])
                isLeft = True
                colors.append(0)
            else:
                total_temp.append(right_temp[i])
                isLeft = False
                first = False
                colors.append(1)
            continue
        else:
            if isLeft:
                if left_temp[i] > 0 and right_temp[i] > 0:
                    total_temp.append((left_temp[i] + right_temp[i]) / 2)
                    # isLeft = True
                    colors.append(2)
                    # count = i
                elif left_temp[i] > 0 and right_temp[i] == 0:
                    first = False
                    total_temp.append(left_temp[i])
                    isLeft = True
                    colors.append(0)
                else:
                    isLeft = False
            else:
                if left_temp[i] > 0 and right_temp[i] > 0:
                    total_temp.append((left_temp[i] + right_temp[i]) / 2)
                    # isLeft = True
                    colors.append(2)
                else:
                    total_temp.append(right_temp[i])
                    colors.append(1)
                # print(left_temp[count], right_temp[count], count)
                # total_temp.append(right_temp[i]+(left_temp[count]-right_temp[count]))
                # total_temp.append(right_temp[i] + (left_temp[count - 1] - right_temp[count - 1]))  #


                # if right_temp[i] < 0:
                #     print(f'i为——{i}')
                #     try:
                #         total_temp[i - 1] = left_temp[i]
                #     except:
                #         print("IndexError: list assignment index out of range")
                #     isLeft = True

    window_size = 3  # 定义窗口大小
    smoothed_temp = np.convolve(total_temp, np.ones(window_size) / window_size, mode='valid')
    x = np.linspace(0, len(smoothed_temp), len(smoothed_temp))
    print(len(smoothed_temp))

    return x, smoothed_temp, colors


if __name__ == '__main__':
    folder_path = r'C:/Users/zmz/Desktop/show_best1'
    for subdir in next(os.walk(folder_path))[1]:
        subdir_path = os.path.join(folder_path, subdir)
        print(f'subdir_path——{subdir_path}')
        left_temp, right_temp, mmath = read_and_sort_images(subdir_path)

        left_zero = left_temp  # 处理之前
        right_zero = right_temp  # 用于绘制三色图

        left_temp_deal = []
        right_temp_deal = []
        # 设定非变化值
        print(left_temp)

        for temp in left_temp:
            if left_temp[0] == 0:
                current_value = 0
            if temp != 0:
                current_value = temp
            left_temp_deal.append(current_value)

        for temp in right_temp:
            if right_temp[0] == 0:
                current_value = 0
            if temp != 0:
                current_value = temp
            right_temp_deal.append(current_value)

        left_temp = left_temp_deal
        right_temp = right_temp_deal

        # 输出结果验证
        print("Max prefix number (mmath):", mmath)
        print("Left temperatures:", left_temp)
        print("Right temperatures:", right_temp)

        # mmath = len(left_temp)  # 假设这是你的mmath值
        print(f"mmath是————————————————————{mmath}")

        # filled_left_temp = fill_zeros_with_average(left_temp)
        # filled_right_temp = fill_zeros_with_average(right_temp)

        # 打印处理后的列表
        # print("Filled left temperatures:", filled_left_temp)
        # print("Filled right temperatures:", filled_right_temp)
        #
        # left_temp = filled_left_temp
        # right_temp = filled_right_temp

        total_temp = []

        """
        # 遍历两个列表的元素
        for left, right in zip(left_temp, right_temp):
            # 如果两个值都不为0，计算平均值
            if left != 0 and right != 0:
                total_temp.append((left + right) / 2)
            # 如果left为0，取right的值
            elif left == 0:
                total_temp.append(right)
            # 如果right为0，取left的值
            else:
                total_temp.append(left)
        
        这是原先那种方法
        """
        #
        # diff = 0
        # for i in range(len(left_zero)):
        #     if right_temp[i] == 0 and left_zero != 0:
        #         total_temp.append(left_zero[i])
        #
        #     elif left_zero[i] == 0 and right_zero != 0:
        #         total_temp.append(right_zero[i])
        #
        #     elif right_zero[i] != 0 and left_zero != 0:
        #         if diff == 0:
        #             diff = left_zero[i] - (left_zero[i] + right_zero[i]) / 2
        #         total_temp.append((right_zero[i] + left_zero[i]) / 2 + diff)
        # 注释掉，采用新的封装好的函数

        x, smoothed_temp, colors = pinjie_LR(left_zero, right_zero)
        plt.figure(figsize=(10, 5))
        for i in range(1, len(colors)):
            if colors[i] == 0:
                cls = 'r'
            elif colors[i] == 1:
                cls = 'b'
            else:
                cls = 'g'
            # cls = 'r' if colors[i] == 0 else 'b'
            plt.plot(x[i - 1: i + 1], smoothed_temp[i - 1:i + 1], cls)
        plt.title('Temperature Curve:LOWESS Smoothed')
        plt.xlabel('Frames')
        plt.ylabel('Temperature (°C)')

        plt.grid(True)
        pp = numPeak(smoothed_temp)[2]
        stamp = time.time()
        plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_pinghua_{pp}.png', dpi=300)
        # plt.close('all')  # 避免内存泄漏

        # total_temp = pinjie_LR(left_temp, right_temp, total_temp)[1]
        # color = pinjie_LR(left_temp, right_temp, total_temp)[0]

        # print(total_temp)

        x = np.linspace(0, len(total_temp), len(total_temp))  # 生成 0 到 10 的 100 个点

        # plt.plot(x[:midpoint], y[:midpoint], 'r')  # 'r' 表示红色
        # 绘制走势图
        # print(total_temp)

        # ================================= #
        # =============================================================================================== #
        # # 转换为numpy数组
        # data = np.array(total_temp)
        # # 时间序列（假设每个数据点代表一个时间单位）
        # x_data = np.arange(len(data))
        # # 应用LOESS平滑
        # frac = 0.07  # 平滑参数，您可以调整这个参数以观察不同的平滑效果
        # smoothed_data_lowess = lowess(total_temp, x_data, frac=frac, return_sorted=False)
        #
        # window_size = 5  # 窗口大小
        # smoothed_data_ma = moving_average(data, window_size)
        #
        # # 绘制原始数据、移动平均和LOESS平滑后的数据
        # plt.figure(figsize=(15, 7))
        #
        # for i in range(1, len(color)):
        #     cls = 'r' if color[i] == 0 else 'b'
        #     plt.plot(x[i - 1:i + 1], total_temp[i - 1:i + 1], cls)
        #
        # plt.xlabel('Frame')
        # plt.ylabel('Temp')
        # plt.title('Temp frame')
        # # ================================= #
        # print(x[0])
        # for i in range(1, len(total_temp)):
        #     cls = 'r' if total_temp[i] == 0 else 'b'
        #     plt.plot(x[i - 1:i + 1], total_temp[i - 1:i + 1], cls)
        # plt.xlabel('Frame')
        # plt.ylabel('Temp')
        # plt.title('Temp frame')
        #
        # pp = numPeak(smoothed_data_lowess)[2]
        # plt.legend()
        # plt.grid(True)

        # pp = 11
        # stamp = time.time()
        # plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_pinghua_{pp}.png', dpi=300)
        # plt.close('all')  # 避免内存泄漏

        """
        # =============================================================================================== #
        # 转换为numpy数组
        data = np.array(total_temp)
        # 时间序列（假设每个数据点代表一个时间单位）
        x_data = np.arange(len(data))
        # 应用LOESS平滑
        frac = 0.07  # 平滑参数，您可以调整这个参数以观察不同的平滑效果
        smoothed_data_lowess = lowess(total_temp, x_data, frac=frac, return_sorted=False)

        window_size = 3  # 窗口大小
        smoothed_data_ma = moving_average(data, window_size)
        
        # 绘制原始数据、移动平均和LOESS平滑后的数据
        plt.figure(figsize=(10, 5))
        # plt.plot(x_data, data, 'k', label='Original Data', alpha=0.3)
        # plt.plot(x_data, smoothed_data_ma, 'b', label='Moving Average Smoothed Data', linewidth=2)
        plt.plot(x_data, smoothed_data_lowess, 'g', label='total_data_LOWESS_Smoothed', linewidth=2)
        pp = numPeak(smoothed_data_lowess)[2]
        plt.title('Temperature Curve:LOWESS Smoothed')
        plt.xlabel('Frames')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        stamp = time.time()
        plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_pinghua_{pp}.png', dpi=300)
        plt.close('all')  # 避免内存泄漏
        # plt.show()
        # =============================================================================================== #
        """
        # 创建一个温度指数的列表，代表每个温度值的索引，用于图表的x轴
        plt.figure(figsize=(10, 5))  # 可以调整画图的大小
        temperature_indices = list(range(mmath))
        # print(f"temperature_indices是————————————————————{len(temperature_indices)}")
        # 使用matplotlib画图
        # 画出left_temp的曲线
        plt.plot(temperature_indices, left_temp, label='Left Temperature', linestyle='-', marker='o')  # , marker='o'
        # 画出right_temp的曲线
        plt.plot(temperature_indices, right_temp, label='Right Temperature', linestyle='-', marker='o')  # , marker='x'
        # 画出total_temp的曲线

        # plt.plot(temperature_indices, total_temp, label='Total Temperature', linestyle='-')  # , marker='*'
        # 添加图例
        plt.legend()

        # 添加图表的标题和轴标签
        plt.title('Temperature Curves of Left and Right Sides')
        plt.xlabel('Frames')
        plt.ylabel('Temperature(℃)')

        # 可选：添加网格线
        plt.grid(True)
        plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_origin.png', dpi=300)
        plt.close('all')  # 避免内存泄漏
        # plt.show()
