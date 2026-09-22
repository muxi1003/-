import os
import re
import cv2
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def load_model(func):
    clf_model = joblib.load(
        '../getRandomForestRegress\clf_model_RGB_0312.pkl')
        # r"C:\cxc\code\python\project_all\yoloV8\temperature_extraction\getRandomForestRegress\clf_model_RGB_1214.pkl")

    def wrapper(img_path, threshold_value=20):
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
            prediction = clf_model.predict(pixel)
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
    total_temp = [0] * mmath
    # right_temp = [0] * mmath
    # 遍历排序后的图片列表
    for image_name, prefix_number in images:
        prefix_number = prefix_number - 1
        # print(prefix_number)
        # 计算图片温度
        image_path = os.path.join(folder_path, image_name)
        temperature = calcTemp(image_path)
        # 根据图片名称中的尾标填充对应的温度列表
        total_temp[prefix_number] = temperature
    return total_temp


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
    peaks, _ = find_peaks(data1, height=None, threshold=None, distance=6,
                          prominence=0.035, width=1, wlen=None,
                          plateau_size=1)
    fenzi = len(peaks)
    fenmu = (fenmu / 8.6) / 60
    respiratory_rate = fenzi / fenmu
    return peaks, respiratory_rate, fenzi


def pinjie_LR_double_nostril(left_zero1, right_zero1):
    left_temp = left_zero1
    right_temp = right_zero1
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
        left_temp = [(temp - non_zero_min_l) / (max_temp_l - non_zero_min_l) if temp - non_zero_min_l > 0 else 0 for
                     temp in left_temp]
    except:
        print("hhhhh_left")
    try:
        right_temp = [(temp - non_zero_min_r) / (max_temp_r - non_zero_min_r) if temp - non_zero_min_r > 0 else 0 for
                      temp in right_temp]
    except:
        print("hhhhh_right")
    colors = []
    total_temp = []
    for i in range(len(left_temp)):
        if left_temp[i] > 0 and right_temp[i] > 0:
            # 策略一、两边均存在，取最平均值
            # total_temp.append((left_temp[i] + right_temp[i]) / 2)
            # 策略二、两边均存在，取最大值、
            total_temp.append(max(left_temp[i], right_temp[i]))
            # 策略三、两边均存在，取最小值
            # total_temp.append(min(left_temp[i] ,right_temp[i]))
            colors.append(2)
        elif left_temp[i] > 0 and right_temp[i] == 0:
            total_temp.append(left_temp[i])
            colors.append(0)
        elif left_temp[i] == 0 and right_temp[i] > 0:
            total_temp.append(right_temp[i])
            colors.append(1)
        else:
            total_temp.append(0.5)
            colors.append(3)
    # https://blog.csdn.net/m0_51233386/article/details/129877889滑动平均滤波
    window_size = 3
    smoothed_temp = np.convolve(total_temp, np.ones(window_size) / window_size, mode='valid')
    x = np.linspace(0, len(smoothed_temp), len(smoothed_temp))
    return x, smoothed_temp, colors


if __name__ == '__main__':
    folder_path = r'G:/l78zdata/0313NEWDATE/last_data_15_double'
    subdir_all = next(os.walk(folder_path))[1]

    df = pd.DataFrame(columns=['subdir', 'pp'])
    for subdir in tqdm(subdir_all, desc='Processing'):
        subdir_path = os.path.join(folder_path, subdir)
        total_temp = read_and_sort_images(subdir_path)

        # ================================================ #
        window_size = 3
        total_temp = np.convolve(total_temp, np.ones(window_size) / window_size, mode='valid')
        # x = np.linspace(0, len(smoothed_temp), len(smoothed_temp))
        # ================================================ #
        print(total_temp)
        temperature_indices = list(range(len(total_temp)))
        ppp = numPeak(total_temp)[0]
        ppp = ppp.astype(int)
        pp = numPeak(total_temp)[2]
        plt.figure(figsize=(10, 5))
        plt.plot(temperature_indices, total_temp, label='total Temperature', linestyle='-')  # , marker='o'
        for i in range(len(ppp)):
            plt.scatter(temperature_indices[ppp[i]], total_temp[ppp[i]], color='k')

        # plt.scatter(ppp, total_temp[ppp], color='k')
        data = {'subdir': subdir, 'pp': pp}
        df = df.append(data, ignore_index=True)
        stamp = time.time()
        plt.title('Temperature Curves of Left and Right Sides')
        plt.xlabel('Frames')
        plt.ylabel('Temperature(℃)')

        # 可选：添加网格线
        plt.grid(True)
        plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_pinghua_{pp}.png', dpi=300)
        plt.close('all')  # 避免内存泄漏

        """
        left_zero = left_temp  # 处理之前
        right_zero = right_temp  # 用于绘制三色图
        left_temp_deal = []
        right_temp_deal = []
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
        total_temp = []

        """
        # # 遍历两个列表的元素
        # for left, right in zip(left_temp, right_temp):
        #     # 如果两个值都不为0，计算平均值
        #     if left != 0 and right != 0:
        #         total_temp.append((left + right) / 2)
        #     # 如果left为0，取right的值
        #     elif left == 0:
        #         total_temp.append(right)
        #     # 如果right为0，取left的值
        #     else:
        #         total_temp.append(left)
        #
        # 这是原先那种方法
        """

        # 20240116下午5:20注解掉，单双鼻孔交替
        x, smoothed_temp, colors = pinjie_LR_double_nostril(left_zero, right_zero)

        ppp = numPeak(smoothed_temp)[0]
        plt.figure(figsize=(10, 5))
        for i in range(1, len(colors)):
            if colors[i] == 0:
                cls = 'r'
            elif colors[i] == 1:
                cls = 'b'
            elif colors[i] == 2:
                cls = 'g'
            else:
                cls = 'k'
            plt.plot(x[i - 1: i + 1], smoothed_temp[i - 1:i + 1], cls)
            # plt.plot(pp, smoothed_temp[pp], 'o', color='y')
            plt.scatter(ppp, smoothed_temp[ppp], color='k')

        pp = numPeak(smoothed_temp)[2]
        plt.title('Temperature Curve:LOWESS Smoothed')
        plt.xlabel('Frames')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)

        data = {'subdir': subdir, 'pp': pp}
        df = df._append(data, ignore_index=True)
        stamp = time.time()
        plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_pinghua_{pp}.png', dpi=300)

        x = np.linspace(0, len(smoothed_temp), len(smoothed_temp))  # 生成 0 到 10 的 100 个点
        # 创建一个温度指数的列表，代表每个温度值的索引，用于图表的x轴
        plt.figure(figsize=(10, 5))  # 可以调整画图的大小
        temperature_indices = list(range(mmath))
        plt.plot(temperature_indices, left_temp, label='Left Temperature', linestyle='-', marker='o')  # , marker='o'
        # 画出right_temp的曲线
        plt.plot(temperature_indices, right_temp, label='Right Temperature', linestyle='-', marker='o')  # , marker='x'
        plt.legend()

        # 添加图表的标题和轴标签
        plt.title('Temperature Curves of Left and Right Sides')
        plt.xlabel('Frames')
        plt.ylabel('Temperature(℃)')

        # 可选：添加网格线
        plt.grid(True)
        plt.savefig(f'{folder_path}/{subdir}_{int(stamp)}_origin.png', dpi=300)
        plt.close('all')  # 避免内存泄漏
        """
    df.to_excel(f'{folder_path}/data_20_total.xlsx', index=False)

