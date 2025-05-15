import glob
import os
import json
import shutil
from sklearn.model_selection import train_test_split

# 是否将关键点可见性信息写入标注
keypoint_is_visual = False
# 类别映射（YOLO格式从0开始）
bbox_class = {'red_leaf': 0, 'blue_leaf': 1}
# 关键点类别顺序
keypoint_class = ['1', '2', '3', '4', 'R']


def process_single_json(json_file, save_label_dir, save_image_dir):
    """处理单个Labelme JSON文件并保存YOLO格式"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像信息
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    image_name = data['imagePath']
    image_src = os.path.join(dataset_root, image_name)

    # 创建YOLO标注内容
    yolo_lines = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # 解析边界框
            label = shape['label']
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)

            # YOLO格式归一化坐标
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 收集关联关键点
            keypoints = {}
            for kp in data['shapes']:
                if kp['shape_type'] == 'point':
                    kp_x, kp_y = kp['points'][0]
                    keypoints[kp['label']] = (kp_x, kp_y)

            # 构建关键点数据
            kp_data = []
            for cls in keypoint_class:
                if cls in keypoints:
                    kx = keypoints[cls][0] / img_width
                    ky = keypoints[cls][1] / img_height
                    kp_data.extend([f"{kx:.5f}", f"{ky:.5f}"])
                    if keypoint_is_visual:
                        kp_data.append("2")  # 可见性
                else:
                    kp_data.extend(["0.00000", "0.00000"])
                    if keypoint_is_visual:
                        kp_data.append("0")

            # 拼接YOLO行
            line = [str(bbox_class[label]), f"{x_center:.5f}", f"{y_center:.5f}",
                    f"{width:.5f}", f"{height:.5f}"] + kp_data
            yolo_lines.append(" ".join(line))

    # 保存YOLO标签文件
    txt_name = os.path.splitext(image_name)[0] + '.txt'
    txt_path = os.path.join(save_label_dir, txt_name)
    with open(txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))

    # 复制图像文件
    if os.path.exists(image_src):
        shutil.copy(image_src, os.path.join(save_image_dir, image_name))
    else:
        print(f"警告：图像文件 {image_src} 不存在")


if __name__ == '__main__':
    # 数据集根目录配置
    dataset_root = r"G:\yolov8\datasets\labelme"
    labelme_dir = os.path.join(dataset_root, "annotations")

    # 创建输出目录结构
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dataset_root, folder), exist_ok=True)

    # 获取并划分数据集
    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))
    train_files, val_files = train_test_split(json_files, test_size=0.18, random_state=42)

    # 处理训练集
    for json_file in train_files:
        try:
            process_single_json(json_file,
                                os.path.join(dataset_root, "labels/train"),
                                os.path.join(dataset_root, "images/train"))
        except Exception as e:
            print(f"处理失败 {json_file}: {str(e)}")

    # 处理验证集
    for json_file in val_files:
        try:
            process_single_json(json_file,
                                os.path.join(dataset_root, "labels/val"),
                                os.path.join(dataset_root, "images/val"))
        except Exception as e:
            print(f"处理失败 {json_file}: {str(e)}")

    print("数据集转换完成！")
    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}")