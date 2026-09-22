import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fftpack import fft

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# 假设这是你的温度列表
temp_list = [35.66655460044689, 35.2881714688512, 35.24755279157222, 35.17343502829035, 35.030665914181355, 35.08906868206386, 35.08771339588502, 35.538048345786144, 35.633659604147596, 35.6652661254879, 35.705672536278115, 35.74288237538362, 35.68088314388301, 35.31476624372402, 35.192203981165946, 35.118149412827265, 35.0982487263487, 35.30540337090653, 35.539943304954946, 35.61793112051152, 35.723492375946925, 35.76436894816087, 35.79554903493372, 35.65209716891964, 35.464578928576216, 35.37828887031138, 35.20886428347561, 35.2253804310822, 35.399551557993185, 35.83880693095317, 35.85356723185225, 35.87066705724426, 35.85003682463051, 35.69295440669325, 35.467122622538184, 35.40314843936224, 35.38216163566768, 35.47214150991802, 35.398707324964064, 35.34136271158199, 35.26459252199343, 35.15504987576011, 35.214945470709154, 35.71895283717777, 35.72282138054609, 35.7348604346112, 35.79705482312552, 35.73461857625586, 35.565947540866325, 35.459664883353085, 35.32781983974928, 35.19683562920562, 35.21579655622136, 35.43384287756675, 35.65516828297419, 35.669940987845266, 35.72800344709395, 35.780605315770856, 35.7747477124864, 35.57649298667257, 35.340960624585065, 35.25573531313465, 35.199764581679545, 35.37623041386586, 35.60077173623675, 35.636935091642414, 35.65891200843731, 35.68558211589253, 35.708682266679986, 35.495118971811394, 35.40455700217993, 35.252960234248405, 35.209599913319245, 35.25346077405837, 35.43410256038217, 35.615468105389326, 35.67755325500045, 35.72770327247889, 35.767307268904304, 35.76684755442821, 35.406704254404005, 35.33414691695075, 35.19095847602809, 35.198392473986395, 35.31292933759734, 35.76386219087661, 35.78264474837481, 35.84315577288647, 35.85716054336078, 35.79877313875325, 35.484666199634674, 35.41915349236183, 35.406107656061934, 35.42029700116831, 35.76731719309774, 35.78250835713682]

# 创建与温度值列表相同长度的索引数组
indices = np.arange(len(temp_list))

# 进行傅里叶变换，找到频率估计
fft_result = np.fft.fft(temp_list)
frequencies = np.fft.fftfreq(len(temp_list))
peaks, _ = find_peaks(np.abs(fft_result), height=0.5)
print(f"fff{peaks}")
estimated_frequency = np.abs(frequencies[peaks][0])
print(estimated_frequency)
# 定义正弦曲线拟合函数
def sine_function(x, amplitude, phase, offset):
    frequency = 2 * np.pi * estimated_frequency
    return amplitude * np.sin(frequency * x + phase) + offset

# 使用 curve_fit 函数进行拟合
initial_guess = [1.0, 0.0, 35.0]  # 不再需要频率初始猜测
params, covariance = curve_fit(sine_function, indices, temp_list, p0=initial_guess)

# 生成拟合的正弦曲线
fit_curve = sine_function(indices, *params)

# 绘制原始数据和拟合的正弦曲线
plt.plot(indices, temp_list, 'o', label='原始数据')
plt.plot(indices, fit_curve, label='正弦曲线拟合')

# 显示图形
plt.legend()
plt.show()

# # 将列表转换为numpy数组
# data = np.array(total_temp)
#
# # 定义移动平均函数
# def moving_average(data, window_size):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # 定义正弦函数模型
# def sine_model(x, amplitude, frequency, phase, offset):
#     return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset
#
# # 应用移动平均去噪
# window_size = 5  # 窗口大小可以根据数据的特点调整
# smoothed_data = moving_average(data, window_size)
#
# # 创建去噪后的数据对应的时间数组
# x_data = np.arange(len(data))
# x_smoothed = np.arange(window_size//2, len(data) - window_size//2)
#
# # 估计数据的周期性
# N = len(smoothed_data)  # 去噪后数据点的数量
# estimated_frequency = 10/N  # 估计频率
#
# # 使用非线性最小二乘法拟合正弦函数模型
# # 注意: 我们使用去噪后的数据和对应的时间数组
# params, covariance = curve_fit(
#     sine_model,
#     x_smoothed,
#     smoothed_data,
#     p0=[np.std(smoothed_data), estimated_frequency, 0, np.mean(smoothed_data)]
# )
#
# # 使用拟合参数生成平滑的正弦波形
# amplitude, frequency, phase, offset = params
# x_fit = np.linspace(x_smoothed[0], x_smoothed[-1], 10000)
# y_fit = sine_model(x_fit, amplitude, frequency, phase, offset)
#
# # 绘制原始数据、去噪后的数据和拟合后的正弦波
# plt.figure(figsize=(12, 6))
# plt.plot(x_data, data, 'o', label='Original Data')
# plt.plot(x_smoothed, smoothed_data, 's', label='Smoothed Data')
# plt.plot(x_fit, y_fit, '-', label='Sine Fit', linewidth=2)
# plt.title('Temperature Data: Original, Smoothed, and Sine Fit')
# plt.xlabel('Time')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.grid()
# plt.show()