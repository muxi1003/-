import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # ensemble集成算法模块，导入随机森林回归模型
from sklearn.model_selection import train_test_split  # 交叉验证
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 回归树衡量分枝质量的指标
# from sklearn.externals import joblib
import joblib


def calcTemp(img, threshold_value=25.953806100238968):
    img1 = cv2.imread(os.path.join(img))
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    x_test = img1.reshape(-1, 1)
    clf_model = joblib.load("../getRandomForestRegress/clf_model_RGB_1214.pkl")
    predictions = clf_model.predict(x_test)
    filetered_value = predictions[predictions > threshold_value]
    average = np.mean(filetered_value)
    return average


if __name__ == '__main__':
    pic_path = ''
    calcTemp(pic_path)
    video_path = r"C:\Users\zmz\Documents\WeChat Files\wxid_6ztvj4zk1p9722\FileStorage\Video\2023-09\a22de15eacbe0d36e3f2f9310b28af93.mp4"
    frame_count = 1
    cap = cv2.VideoCapture(video_path)
    while cap:
        success, frame = cap.read()
        if success:
            cv2.imwrite(f"./pic1/{frame_count}.jpg", frame)
            frame_count += 1
        else:
            break
    cap.release()
