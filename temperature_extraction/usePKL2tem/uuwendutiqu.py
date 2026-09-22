import os
import sys
import cv2
import time
import joblib
import numpy as np
from optparse import OptionParser


def load_model(func):
    clf_model = joblib.load(r"C:\cxc\code\python\project_all\yoloV8\temperature_extraction\getRandomForestRegress\clf_model_RGB_1214.pkl")

    def wrapper(img_path, img, threshold_value=30):
        img1 = cv2.imread(os.path.join(img_path, img))
        x_test = img1.reshape(-1, 3)
        predictions = clf_model.predict(x_test)
        filetered_value = predictions[predictions > threshold_value]
        average = np.mean(filetered_value)
        return average

    return wrapper

@load_model
def calcTemp(img_path, img, clf_model, threshold_value=30):
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
    # x_test = img1.reshape(-1, 3)
    # predictions = clf_model.predict(x_test)
    # filetered_value = predictions[predictions > threshold_value]
    # average = np.mean(filetered_value)
    # return average

def process_image(img_path, img):
    result = calcTemp(img_path, img)
    print(f'图片 {img} 计算完成')
    return result


def option_input(argv):
    (options, args) = parser.parse_args(argv)
    return options.path, options.key


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--path', default=r'C:\Users\zmz\Desktop\pic', type=str, dest='path')
    parser.add_option('-k', '--key', default=1, type=str, dest='key')
    excel_file_path = option_input(sys.argv)[0]
    key_guan = int(option_input(sys.argv)[1])
    if key_guan == 1:
        path = excel_file_path
        # 获取 cow_video_frame 文件夹下的子文件夹列表
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        file_num = 1

        for folder in subfolders:
            # 构建 frame_yes 和 frame_no 文件夹的路径
            img_path = os.path.join(folder, 'frame_yes')
            img_list = os.listdir(img_path)
            img_list.sort(key=lambda x: int(x.split("_")[0]))
            tem_list = []
            pic_num = 1
            time1 = time.time()
            print(img_list)
            for img in img_list:
                single_tem = calcTemp(img_path, img)
                tem_list.append(single_tem)
                print(f'第{file_num}个视频，编号为{folder}的视频中第{pic_num}张图片{single_tem}计算完成。')
                pic_num += 1
            file_num += 1
        print(tem_list)
        sys.exit()