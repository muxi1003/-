from ultralytics import YOLO
import pandas as pd

# Sanity check: ensure we imported the real pandas, not a local shadow file
assert hasattr(pd, 'DataFrame'), (
    f'Imported "pandas" from {pd.__file__}, which lacks DataFrame. '
    'Check for a local pandas.py file shadowing the real package.'
)

if __name__ == '__main__':
    # 定义要对比的模型列表
    models_to_test = {
        'YOLOv8n': r'E:\real\use_code\yoloV8\yolov8n-pose.pt',
        'YOLO11n': r'E:\real\use_code\yoloV8\yolo11n-pose.pt',  # 2024年底发布的版本
        'YOLO26n': r'E:\real\use_code\yoloV8\yolo26n-pose.pt'  # 最新专为边缘计算优化的端到端版本
    }

    yaml_path = r'E:\real\use_code\yoloV8\Dataset_new\72video\dataset.yaml'
    results_data = []

    for name, pt_file in models_to_test.items():
        print(f"========== 正在训练与评估 {name} ==========")
        model = YOLO(pt_file)

        # 统一训练 100 轮进行对比
        model.train(data=yaml_path, epochs=100, batch=8, imgsz=640, name=f'{name}_cow_pose')

        # 获取验证集精度和速度
        metrics = model.val()
        map50 = metrics.pose.map50
        map50_95 = metrics.pose.map
        speed_ms = metrics.speed['inference']  # 推理耗时 (毫秒)

        results_data.append([name, map50, map50_95, speed_ms])

    # 保存对比结果，直接用作论文图表！
    df = pd.DataFrame(results_data, columns=['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Inference Speed(ms)'])
    df.to_csv('model_comparison_results.csv', index=False)
    print(df)