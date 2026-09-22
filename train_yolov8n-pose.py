from ultralytics import YOLO
import os
import logging

# -------------------------- 日志配置（便于排查训练问题）--------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo11_train_log.txt', encoding='utf-8'),  # 日志保存到文件
        logging.StreamHandler()  # 日志打印到控制台
    ]
)
logger = logging.getLogger(__name__)

# -------------------------- 核心参数配置（可根据硬件调整）--------------------------
# 模型配置
MODEL_PATH = r'E:\real\use_code\yoloV8\yolo11n-pose.pt'  # 预训练模型路径
MODEL_SAVE_NAME = 'cow_nose_pose'  # 训练后模型保存名称
SAVE_DIR = 'runs/pose'  # 模型保存根目录（自动创建）

# 数据集配置（与splitDataset.py生成的yaml文件路径一致）
DATASET_YAML = r'E:\real\use_code\yoloV8\Dataset_new\72video\dataset.yaml'

# 训练超参数
EPOCHS = 200  # 训练轮数，50轮可满足精度要求，若mAP不达标可增至80轮
BATCH_SIZE = 8  # 批次大小，GPU显存≥4G用8，显存不足设为4或2
IMG_SIZE = 640  # 输入图像分辨率，与红外图像实际分辨率匹配
DEVICE = 0  # 训练设备：0=GPU，-1=CPU，多个GPU用[0,1]
PATIENCE = 50  # 早停策略，10轮无精度提升则停止训练，避免过拟合
LEARNING_RATE = 0.001  # 学习率，默认即可，无需调整
WEIGHT_DECAY = 0.0005  # 权重衰减，防止过拟合


# -------------------------- 模型训练核心逻辑--------------------------
def train_yolo11_nose_pose():
    try:
        # 1. 检查数据集配置文件是否存在
        if not os.path.exists(DATASET_YAML):
            logger.error(f"数据集配置文件不存在！路径：{DATASET_YAML}")
            logger.error("请先执行pre_data/all_splitDataset.py生成dataset.yaml文件")
            raise FileNotFoundError(f"Dataset yaml file not found: {DATASET_YAML}")

        # 2. 加载YOLO11n-Pose模型
        logger.info("开始加载YOLO11n-Pose预训练模型...")
        model = YOLO(MODEL_PATH)
        logger.info("预训练模型加载成功！")

        # 3. 开始模型训练
        logger.info("=" * 50)
        logger.info("开始训练奶牛鼻孔关键点检测模型")
        logger.info(f"训练参数：epochs={EPOCHS}, batch={BATCH_SIZE}, imgsz={IMG_SIZE}, device={DEVICE}")
        logger.info("=" * 50)

        results = model.train(
            data=DATASET_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            name=MODEL_SAVE_NAME,
            save=True,
            patience=PATIENCE,
            lr0=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            verbose=True,  # 打印训练过程详情
            project=SAVE_DIR,  # 模型保存项目目录
            exist_ok=True  # 允许覆盖已有训练结果
        )

        # 4. 模型验证（训练完成后自动执行，输出精度指标）
        logger.info("训练完成，开始模型验证...")
        val_results = model.val()

        # 评估关键点（Pose）的精度
        pose_map50 = val_results.pose.map50
        logger.info(f"模型验证结果：Pose mAP@0.5={pose_map50:.4f}")


        # 5. 校验核心指标是否达标
        if val_results.pose.map50 >= 0.994:
            logger.info("模型训练达标！mAP@0.5≥0.995，满足鼻孔定位要求")
        else:
            logger.warning("模型训练未达标！mAP@0.5<0.995，建议调整参数重新训练")
            logger.warning("调整建议：1. 增加训练轮数至80；2. 补充标注数据；3. 调整学习率为0.0005")

        # 6. 【修改点】去除多余的 export，直接提示模型位置
        best_model_path = os.path.join(SAVE_DIR, MODEL_SAVE_NAME, 'weights', 'best.pt')
        logger.info(f"模型已自动保存！最佳模型路径：{best_model_path}")
        logger.info("建议将 best.pt 复制至 models 文件夹，方便后续调用")

        return results, val_results

    except Exception as e:
        logger.error(f"模型训练失败！错误信息：{str(e)}", exc_info=True)
        raise  # 抛出异常，便于排查问题


# -------------------------- 执行训练--------------------------
if __name__ == "__main__":
    try:
        train_yolo11_nose_pose()
    except Exception as e:
        print(f"训练终止，错误原因：{str(e)}")
        print("请查看yolo11_train_log.txt日志文件，获取详细错误信息")
