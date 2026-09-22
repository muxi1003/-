一、数据集处理
先运行extract_frames.py对热红外视频进行提取帧，提取完帧后用labelme进行标注，标注一个鼻子框和两个鼻子关键点：左鼻子和右鼻子。
标注完后，运行labelme2YOLO_batch.py将labelme格式.json文件转换为YOLO训练的格式.txt文件。
可以运行auto_viewLabel.py查看标注是否正确。
转换完标注文件后，运行all_splitDataset.py将图片文件和标注文件进行同一分类为train和val。

二、训练YOLOV8n-pose模型
分类完后运行train_yolov8n-pose.py训练yolov8n-pose标注鼻子关键点得到best.pt，用best.pt验证模型推理效果
在命令行运行yolo pose predict model=runs/pose/cow_nose_pose/weights/best.pt source=你的原视频路径.mp4 show=True，观察：两个点是不是稳稳地“贴”在牛的左鼻孔和右鼻孔上。

三、RGB转换为温度值
温度映射机制（RGB 转 温度）：运行yoloV8\temperature_extraction\getRandomForestRegress\RGB_nihe.py训练出一个小型的随机森林模型.pkl文件

四、提取双侧鼻孔温度变化曲线
/*运行cropped_noses.py可以得到左鼻子右鼻子取半径为20像素的区域。*/
1.运行one_clink_temp.py文件：得到左鼻子右鼻子取半径为20像素的区域。用训练好的随机森林模型.pkl，把这区域里的 RGB 像素变成温度值，算出左鼻孔平均温度和右鼻孔平均温度。
把这些温度保存到一个 CSV 文件里（每一行是：帧号, 左鼻孔温度, 右鼻孔温度）。

五、数据融合与呼吸率检测 (Draw Curve)
1.运行draw_final_curve.py可以得到呼吸曲线

有了 CSV 里的温度序列，就进入了 温度曲线融合（Fusion of temperature curves）和峰值检测。
打开 draw_curve 文件夹
运行 curve.py 或 curve_g.py
这些脚本应该会读取刚才生成的 CSV，执行：Min-Max 归一化（把左右鼻孔温度统一缩放）。融合策略（当两个鼻孔都在时，取两者的最大值）。滑动平均滤波 (MAF)：把毛刺过滤掉，让曲线平滑（像正弦波一样）。寻峰算法 (Peak Detection)：数一数曲线有几个波峰，计算出这段时间的呼吸率 (RR)。
输出： 呼吸率（RR）数值和/或融合后的曲线图。