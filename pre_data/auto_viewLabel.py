# 按“空格键”切换下一张图像，按“ESC”键退出；
import cv2
import numpy as np
import json
import os
import glob

# -------------------------- 配置参数 --------------------------
# 图片文件夹路径
IMG_DIR = r"E:\real\use_code\yoloV8\Dataset_new\images1"
# labelme标注文件文件夹路径
LABELME_DIR = r"E:\real\use_code\yoloV8\Dataset_new\images1"
# 保存可视化结果的文件夹
SAVE_DIR = r"E:\real\use_code\yoloV8\Dataset_new\viewimage"
# 框的可视化配置
bbox_labelstr = {
    'font_size': 6,  # 字体大小
    'font_thickness': 14,  # 字体粗细
    'offset_x': 0,  # x方向，文字偏移距离，向右为正
    'offset_y': -80,  # y方向，文字偏移距离，向下为正
}
bbox_color = (255, 129, 0)
bbox_thickness = 5

# 关键点配色
kpt_color_map = {
    'left_nostril': {'id': 0, 'color': [255, 0, 0], 'radius': 20, 'thickness': -1},
    'right_nostril': {'id': 1, 'color': [0, 255, 0], 'radius': 20, 'thickness': -1},
    # 'angle_90': {'id': 2, 'color': [0, 0, 255], 'radius': 30, 'thickness': -1}
}

# 点类别文字配置
kpt_labelstr = {
    'font_size': 4,  # 字体大小
    'font_thickness': 12,  # 字体粗细
    'offset_x': 30,  # X 方向，文字偏移距离，向右为正
    'offset_y': 100,  # Y 方向，文字偏移距离，向下为正
}


# -------------------------- 工具函数 --------------------------
def get_labelme_path(img_path, labelme_dir):
    """根据图片路径获取对应的labelme标注文件路径"""
    img_name = os.path.basename(img_path)
    labelme_name = img_name.replace('.jpg', '.json')  # 假设图片是jpg格式
    return os.path.join(labelme_dir, labelme_name)


def draw_annotation(img_bgr, labelme_path):
    """绘制标注（框+关键点）"""
    # 读取labelme标注文件
    if not os.path.exists(labelme_path):
        print(f"警告：未找到标注文件 {labelme_path}")
        return img_bgr

    with open(labelme_path, 'r', encoding="utf-8") as f:
        labelme = json.load(f)

    # 绘制矩形框
    for each_ann in labelme["shapes"]:
        if each_ann["shape_type"] == "rectangle":
            bbox_label = each_ann['label']
            bbox_keypoints = each_ann['points']

            # 计算框的左上角和右下角坐标（兼容2点和4点矩形标注）
            bbox_top_left_x = int(min(p[0] for p in bbox_keypoints))
            bbox_top_left_y = int(min(p[1] for p in bbox_keypoints))
            bbox_bottom_right_x = int(max(p[0] for p in bbox_keypoints))
            bbox_bottom_right_y = int(max(p[1] for p in bbox_keypoints))

            # 画矩形框
            img_bgr = cv2.rectangle(img_bgr, (bbox_top_left_x, bbox_top_left_y),
                                    (bbox_bottom_right_x, bbox_bottom_right_y),
                                    bbox_color, bbox_thickness)

            # 写框的类别文字
            img_bgr = cv2.putText(img_bgr, bbox_label, (
                bbox_top_left_x + bbox_labelstr['offset_x'], bbox_top_left_y + bbox_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                                  bbox_labelstr['font_thickness'])

        # 绘制关键点
        if each_ann['shape_type'] == 'point':
            kpt_label = each_ann['label']
            if kpt_label not in kpt_color_map:
                print(f"警告：关键点类别 {kpt_label} 无配色配置，跳过绘制")
                continue

            # 获取关键点坐标
            kpt_xy = each_ann['points'][0]
            kpt_x, kpt_y = int(kpt_xy[0]), int(kpt_xy[1])

            # 获取关键点可视化配置
            kpt_config = kpt_color_map[kpt_label]
            kpt_color = kpt_config['color']
            kpt_radius = kpt_config['radius']
            kpt_thickness = kpt_config['thickness']

            # 画关键点圆
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, kpt_thickness)

            # 写关键点类别文字
            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                  kpt_labelstr['font_thickness'])
    return img_bgr


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 创建保存文件夹（如果不存在）
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 获取所有图片路径（按名称排序，保证顺序）
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))  # 只匹配jpg图片
    if not img_paths:
        print("错误：未找到任何图片！")
        exit(1)

    # 循环遍历每张图片
    for idx, img_path in enumerate(img_paths):
        # 读取图片
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"警告：无法读取图片 {img_path}，跳过")
            continue

        # 获取对应标注文件路径
        labelme_path = get_labelme_path(img_path, LABELME_DIR)

        # 绘制标注
        img_bgr = draw_annotation(img_bgr, labelme_path)

        # 显示图片（调整窗口大小，避免图片过大）
        cv2.namedWindow("Annotation View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotation View", 1280, 720)  # 设置窗口大小为1280x720
        cv2.imshow("Annotation View", img_bgr)

        # 保存可视化结果（可选）
        #save_path = os.path.join(SAVE_DIR, f"frame_{idx:06d}.jpg")
        #cv2.imwrite(save_path, img_bgr)
        #print(f"已显示并保存：{save_path} (第{idx + 1}/{len(img_paths)}张)")

        # 监听键盘事件
        key = cv2.waitKey(0)  # 0表示等待按键输入
        if key == 27:  # ESC键（ASCII码27）：退出程序
            print("按下ESC键，退出程序")
            break
        elif key == 32:  # 空格键（ASCII码32）：切换下一张
            print("按下空格键，切换下一张")
            continue
        else:
            print(f"按下未知按键（{key}），切换下一张（按ESC退出，按空格切换）")
            continue

    # 释放资源
    cv2.destroyAllWindows()
    print("程序结束")