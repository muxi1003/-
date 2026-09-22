import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # ensemble集成算法模块，导入随机森林回归模型
from sklearn.model_selection import train_test_split  # 交叉验证
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 回归树衡量分枝质量的指标
import joblib

# 新增LightGBM（树模型的进化版）
from lightgbm import LGBMRegressor

# 1、将图片的像素值与温度值一一对其，进行读取
y = pd.read_csv('./1.csv', header=None)     # 真实的温度矩阵
img1 = cv2.imread("./1.png")                               # 对应的伪彩色图像

x_np = img1.reshape(img1.shape[0]*img1.shape[1], 3)  # 原图中的RGB
y_np = y.to_numpy().reshape(-1, 1)

# 2、利用sklearn进行拟合
clf = RandomForestRegressor()  # 创建随机森林回归器对象
X_train, X_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.4, random_state=42)
clf.fit(X_train, y_train.ravel())  # 拟合模型

joblib.dump(clf, 'clf_model_RGB_20240906.pkl')
# 3、拟合指标
y_pre = clf.predict(X_test)
r2_score = clf.score(X_test, y_test)  # 回归树的接口返回的是R^2，而不是MSE
MSE = mean_squared_error(y_test, y_pre)
MAE = mean_absolute_error(y_test, y_pre)

print('mean_squared_error', MSE)
print('r2_score', r2_score)
print('MAE', MAE)


"""
# 根据预测的温度值进行对应的复现
pre = clf.predict(x_np)
img_np = pre.reshape(*img1.shape[:-1])
plt.imshow(img_np)
plt.show()
"""

"""
预测结果：
mean_squared_error 0.0009189723523725914
r2_score 0.9999282664227951
MAE 0.016602172898269946
"""
