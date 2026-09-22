import os
import json
import numpy as np
from tqdm import tqdm

# 框的类别
bbox_class = {
    'nose': 0,
}
# 关键点的类别
keypoint_class = ['left_nostril', 'right_nostril']


def process_single_json(labelme_path, save_folder):
    # print(save_folder)
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']  # 图像宽度
    img_height = labelme['imageHeight']  # 图像高度

    # 生成 YOLO 格式的 txt 文件
    yolo_filename = os.path.splitext(os.path.basename(labelme_path))[0] + '.txt'
    yolo_txt_path = os.path.join(save_folder, yolo_filename)


    with open(yolo_txt_path, 'w', encoding='utf-8') as f:

        for each_ann in labelme['shapes']:  # 遍历每个标注

            if each_ann['shape_type'] == 'rectangle':  # 每个框，在 txt 里写一行

                yolo_str = ''

                ## 框的信息
                # 框的类别 ID
                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)
                # 取所有点的 min/max（兼容 2-point 对角 和 4-point 四角两种矩形格式）
                all_x = [p[0] for p in each_ann['points']]
                all_y = [p[1] for p in each_ann['points']]
                bbox_top_left_x = int(min(all_x))
                bbox_bottom_right_x = int(max(all_x))
                bbox_top_left_y = int(min(all_y))
                bbox_bottom_right_y = int(max(all_y))
                # 框中心点的 XY 像素坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                # 框宽度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                # 框高度
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                # 框中心点归一化坐标
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                # 框归一化宽度
                bbox_width_norm = bbox_width / img_width
                # 框归一化高度
                bbox_height_norm = bbox_height / img_height

                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm,
                                                                  bbox_width_norm, bbox_height_norm)

                ## 找到该框中所有关键点，存在字典 bbox_keypoints_dict 中
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']:  # 遍历所有标注
                    if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                        # 关键点XY坐标、类别
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        if (x > bbox_top_left_x) & (x < bbox_bottom_right_x) & (y < bbox_bottom_right_y) & (
                                y > bbox_top_left_y):  # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y]

                ## 把关键点按顺序排好
                for each_class in keypoint_class:  # 遍历每一类关键点
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2)  # 2-可见不遮挡 1-遮挡 0-没有点
                    else:  # 不存在的点，一律为0
                        yolo_str += '0 0 0 '
                # 写入 txt 文件中
                f.write(yolo_str + '\n')
    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))


# with open(labelme_path, 'r', encoding='utf-8') as f:
#     labelme = json.load(f)
#     img_width = labelme['imageWidth']  # 图像宽度
#     img_height = labelme['imageHeight']  # 图像高度
#
#     # 生成 YOLO 格式的 txt 文件
#     suffix = labelme_path.split('.')[-2]
#     yolo_txt_path = suffix + '.txt'
#     print(yolo_txt_path)
#     print(labelme['shapes'])

if __name__ == '__main__':
    save_folder = r'E:\real\use_code\yoloV8\Dataset_new\al_labelyolo\zs16110577'
    labelme_path_total = r"E:\real\use_code\yoloV8\Dataset_new\al_labelme\zs16110577"
    os.makedirs(save_folder, exist_ok=True)
    for labelme_path in os.listdir(labelme_path_total):
        if not labelme_path.lower().endswith('.json'):
            continue
        try:
            process_single_json(os.path.join(labelme_path_total, labelme_path), save_folder=save_folder)
        except Exception as e:
            print('******有误****** {}: {}'.format(labelme_path, e))
    print('YOLO格式的txt标注文件已保存至 ', save_folder)
