import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

img_path = r"E:/Desktop/l78z/data/pic/1.jpg"
img_bgr = cv2.imread(img_path)

bbox_labelstr = {
    'font_size': 6,  # 字体大小
    'font_thickness': 14,  # 字体粗细
    'offset_x': 0,  # x方向，文字偏移距离，向右为正
    'offset_y': -80,  # y方向，文字偏移距离，向下为正
}
# 框的可视化配置
bbox_color = (255, 129, 0)
bbox_thickness = 5

# 关键点配色
kpt_color_map = {
    'left_nostril': {'id': 0, 'color': [255, 0, 0], 'radius': 20, 'thickness': -1},
    'right_nostril': {'id': 1, 'color': [0, 255, 0], 'radius': 20, 'thickness': -1},
    # 'angle_90': {'id': 2, 'color': [0, 0, 255], 'radius': 30, 'thickness': -1}
}

# 点类别文字
kpt_labelstr = {
    'font_size': 4,  # 字体大小
    'font_thickness': 12,  # 字体粗细
    'offset_x': 30,  # X 方向，文字偏移距离，向右为正
    'offset_y': 100,  # Y 方向，文字偏移距离，向下为正
}

labelme_path = r"E:/Desktop/l78z/data/pic/1.json"
with open(labelme_path, 'r', encoding="utf-8") as f:
    labelme = json.load(f)
    # print(labelme.keys())
    print(labelme["shapes"])

    for each_ann in labelme["shapes"]:
        if each_ann["shape_type"] == "rectangle":
            # 框的类别
            bbox_label = each_ann['label']
            # 框的两点坐标
            bbox_keypoints = each_ann['points']
            # 计算框的左上角和右下角坐标（兼容2点和4点矩形标注）
            bbox_top_left_x = int(min(p[0] for p in bbox_keypoints))
            bbox_top_left_y = int(min(p[1] for p in bbox_keypoints))
            bbox_bottom_right_x = int(max(p[0] for p in bbox_keypoints))
            bbox_bottom_right_y = int(max(p[1] for p in bbox_keypoints))

            # 画矩形：画框
            img_bgr = cv2.rectangle(img_bgr, (bbox_top_left_x, bbox_top_left_y),
                                    (bbox_bottom_right_x, bbox_bottom_right_y),
                                    bbox_color, bbox_thickness)

            # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            img_bgr = cv2.putText(img_bgr, bbox_label, (
                bbox_top_left_x + bbox_labelstr['offset_x'], bbox_top_left_y + bbox_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                                  bbox_labelstr['font_thickness'])
            # plt.imshow(img_bgr[:, :, ::-1])
            # plt.show()
        if each_ann['shape_type'] == 'point':  # 筛选出关键点标注

            kpt_label = each_ann['label']  # 该点的类别
            print(kpt_label)

            # 该点的 XY 坐标
            kpt_xy = each_ann['points'][0]
            kpt_x, kpt_y = int(kpt_xy[0]), int(kpt_xy[1])

            # 该点的可视化配置
            kpt_color = kpt_color_map[kpt_label]['color']  # 颜色
            kpt_radius = kpt_color_map[kpt_label]['radius']  # 半径
            kpt_thickness = kpt_color_map[kpt_label]['thickness']  # 线宽（-1代表填充）

            # 画圆：画该关键点
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, kpt_thickness)

            # 写该点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                  kpt_labelstr['font_thickness'])
    # for each_ann in labelme["shapes"]:

    plt.imshow(img_bgr[:, :, ::-1])
    cv2.imwrite("E:/desktop/123333.jpg", img_bgr)
    plt.show()

    pass
