#! python3.8 功能说明;时间2023-10-25可使用，用于计算每张图片的温度值，并且存储于一个excel表格；
# 1->利用pkl温度模型获取温度值 2->温度值存入列表，并保存温度值到excel表格 3->温度值列表进行滤波，之后显示，并且保存图片结果
import os
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

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import concurrent.futures
from optparse import OptionParser
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter


def load_model(func):
    clf_model = joblib.load(r"E:\real\use_code\yoloV8\temperature_extraction\getRandomForestRegress\clf_model_RGB_20240906.pkl")

    def wrapper(img_path, img, threshold_value=30):
        img1 = cv2.imread(os.path.join(img_path, img))
        x_test = img1.reshape(-1, 3)
        predictions = clf_model.predict(x_test)
        filetered_value = predictions[predictions > threshold_value]
        average = np.mean(filetered_value)
        return average

    return wrapper


@load_model
def calcTemp(img_path, img, clf_model):
    img1 = cv2.imread(os.path.join(img_path, img))
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


def process_image(img_path, img):
    result = calcTemp(img_path, img)
    # print(f'图片 {img} 计算完成')
    return result


def temp_save(table_name, tem: list, path1):
    df = pd.DataFrame({'tem_value': tem})
    excel1 = f'{table_name}_{getNowTime()}.xlsx'
    excel1 = os.path.join(path1, excel1)
    df.to_excel(excel1, index=False)


def draw_pic(tem_list: list, path1, table_name):
    desired_distance = 4
    tem_array = np.array(tem_list)
    # gaussian
    sigma = 0.8
    maf = moving_average_filter(tem_array, 3)
    plt.figure(figsize=(12, 6))

    # 1.原始温度数据
    plt.subplot(1, 2, 1)
    plt.plot(tem_array, linestyle='-', color='b')
    plt.title('origin temp')
    plt.xlabel('frame step')
    plt.ylabel('temperature')
    pp = numPeak(tem_array.tolist(), desired_distance)[0]
    yy_ax = np.array(tem_array)
    plt.plot(pp, yy_ax[pp], 'o', label='origin_temp peak', color='g')

    # 2.中值滤波之后的温度数据
    plt.subplot(1, 2, 2)
    plt.plot(maf, linestyle='-', color='r')
    plt.title('maf temp')
    plt.xlabel('frame step')
    plt.ylabel('temperature')
    pp = numPeak(maf.tolist(), desired_distance)[0]
    yy_ax = np.array(maf)
    rate = int(numPeak(maf.tolist(), desired_distance)[1])
    cishu1 = int(numPeak(maf.tolist(), desired_distance)[2])
    plt.plot(pp, yy_ax[pp], 'go', label='maf_temp_peak')
    plt.legend()
    plt.tight_layout(pad=2.0)

    excel1 = f'{table_name + "_" + "yuantushengcheng_" + f"{getNowTime()}_" + str(rate)}.png'
    excel1 = os.path.join(path1, excel1)
    # plt.show()
    plt.savefig(excel1)
    plt.close()
    return cishu1


def moving_average_filter(data, window_size):
    filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return filtered_data


def numPeak(datalist: list, desired_distance: int):
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


def getNowTime():
    current_time = datetime.now()
    year = str(current_time.year)
    month = str(current_time.month).zfill(2)
    day = str(current_time.day).zfill(2)
    hour = str(current_time.hour).zfill(2)
    minute = str(current_time.minute).zfill(2)
    second = str(current_time.second).zfill(2)
    formatted_time = "_".join([year, month, day, hour, minute, second])
    return formatted_time


def sendText(msg, user_phone):
    url = "https://oapi.dingtalk.com/robot/send?access_token" \
          "=03939ac4d1f20df817bfc163400fc77fed4d38039a764d883c9e8c17e3e2d4db "
    message = {
        "msgtype": "text",
        "text": {
            "content": "光头强，你又来砍树了:\n" + msg
        },
        "at": {
            # cxc_add 根据手机号@对应的人
            "atMobiles": [user_phone],
            # cxc_add 是否@所有人
            "isAtAll": False
        }
    }
    requests.post(url, json=message)


def option_input(argv):
    (options, args) = parser.parse_args(argv)
    return options.path, options.key


def get_subfolders(folder_path):
    subfolders = [f.name for f in sorted(os.scandir(folder_path), key=lambda f: f.name.lower()) if f.is_dir()]
    return subfolders


def save_cishuexcel(xuhaoList, cishu, path1):
    output_file = os.path.join(path1, "file.xlsx")
    # output_file = 'G:/file_gg.xlsx'
    wb = openpyxl.Workbook()
    ws = wb.active

    # Write subfolders to a single column
    for i, (subfolder, data) in enumerate(zip(xuhaoList, cishu), start=1):
        ws.cell(row=i, column=1, value=subfolder)
        ws.cell(row=i, column=2, value=data)

    # Save the workbook
    wb.save(output_file)
    pass


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--path', default='D:/l78zdata/cow/0806', type=str, dest='path')
    parser.add_option('-k', '--key', default=1, type=str, dest='key')
    excel_file_path = option_input(sys.argv)[0]
    key_guan = int(option_input(sys.argv)[1])

    if key_guan == 1:
        cishu = []
        xuhaoList = []
        path = excel_file_path
        # 获取 cow_video_frame 文件夹下的子文件夹列表
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        file_num = 1
        for folder in subfolders:
            # 构建 frame_yes 和 frame_no 文件夹的路径
            img_path = os.path.join(folder, 'frame_no')
            img_list = os.listdir(img_path)
            try:
                img_list.sort(key=lambda x: int(x.split(".")[0]))
            except:
                continue
            pic_num = 1
            time1 = time.time()
            # 创建线程池
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 提交任务并获取结果，保证顺序
                futures = [executor.submit(process_image, img_path, img) for img in img_list]
                # 获取计算结果
                tem_list = [future.result() for future in futures]

            temp_save(folder, tem_list, os.path.join(path, folder))
            c = draw_pic(tem_list, os.path.join(path, folder), folder)
            # 输出检测到的呼吸次数=========================================================== #
            xuha1 = folder.split("\\")[-1]
            print(f"文件夹名：{xuha1}——次数：{c}")
            xuhaoList.append(xuha1)
            cishu.append(c)

            # over ======================================================================== #
            if file_num % 10 == 0:
                time2 = time.time()
                a = time2 - time1
                sendText(
                    f'第{file_num}个视频,编号为{folder}的前十个视频处理全部结束' + '总用时{}s'.format(
                        a), 19801567621)
            file_num += 1
        save_cishuexcel(xuhaoList, cishu, os.path.join(path, excel_file_path))
    else:
        raise ValueError("key_guan值不对，请重新选择（1 or 2 or 3）")
    print("ALL over~")
