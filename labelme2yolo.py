import glob
import os
import json
import shutil
from sklearn.model_selection import train_test_split

# 是否将关键点可见性信息写入标注
keypoint_is_visual = False
# 类别映射（YOLO格式从0开始）
bbox_class = {'Ured': 0, 'Ublue': 1}
# 关键点类别顺序
keypoint_class = ['top', 'right', 'bottom', 'left', 'R']


def process_single_json(json_file, save_label_dir, save_image_dir):
    """处理单个Labelme JSON文件并保存YOLO格式，返回处理是否成功"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ========== 新增：基础字段校验 ==========
        required_fields = ['imageWidth', 'imageHeight', 'imagePath', 'shapes']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺失必要字段: {field}")

        # 获取图像信息
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        image_name = data['imagePath']
        image_src = os.path.join(dataset_root, 'img', image_name)  # 修正路径构造

        # ========== 新增：图片存在性校验 ==========
        if not os.path.exists(image_src):
            raise FileNotFoundError(f"关联图片不存在: {image_src}")

        # 创建YOLO标注内容
        yolo_lines = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                # 解析边界框
                label = shape['label']

                # ========== 新增：标签合法性校验 ==========
                if label not in bbox_class:
                    raise ValueError(f"非法标签: {label}")

                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)

                # 坐标范围校验
                if any(val < 0 for val in [x_min, y_min, x_max, y_max]):
                    raise ValueError("坐标值不能为负数")
                if x_max > img_width or y_max > img_height:
                    raise ValueError("坐标超出图像范围")

                # YOLO格式归一化坐标
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # 收集关联关键点
                keypoints = {}
                for kp in data['shapes']:
                    if kp['shape_type'] == 'point':
                        kp_label = kp['label']
                        if kp_label not in keypoint_class:
                            raise ValueError(f"非法关键点标签: {kp_label}")
                        kp_x, kp_y = kp['points'][0]
                        keypoints[kp_label] = (kp_x, kp_y)

                # 关键点完整性校验
                missing_kps = [cls for cls in keypoint_class if cls not in keypoints]
                if missing_kps:
                    raise ValueError(f"缺失关键点: {missing_kps}")

                # 构建关键点数据
                kp_data = []
                for cls in keypoint_class:
                    kx = keypoints[cls][0] / img_width
                    ky = keypoints[cls][1] / img_height
                    kp_data.extend([f"{kx:.5f}", f"{ky:.5f}"])
                    if keypoint_is_visual:
                        kp_data.append("2")  # 可见性

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
        shutil.copy(image_src, os.path.join(save_image_dir, image_name))
        return True

    except (KeyError, json.JSONDecodeError, ValueError, FileNotFoundError) as e:
        print(f"处理失败 {json_file} - 错误类型: {type(e).__name__}, 详情: {str(e)}")
        return False
    except Exception as e:
        print(f"未知错误: {json_file} - {str(e)}")
        return False


def delete_invalid_files(json_file):
    """安全删除无效文件"""
    try:
        # 获取关联图片路径
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image_name = data.get('imagePath', '')
        image_path = os.path.join(dataset_root, 'img', image_name)

        # 删除JSON文件
        if os.path.exists(json_file):
            os.remove(json_file)
            print(f"已删除无效JSON: {json_file}")

        # 删除关联图片
        if image_name and os.path.exists(image_path):
            os.remove(image_path)
            print(f"已删除关联图片: {image_path}")

    except Exception as e:
        print(f"删除操作失败: {json_file} - {str(e)}")


if __name__ == '__main__':
    # 数据集根目录配置
    dataset_root = r"E:\newrune"
    labelme_dir = os.path.join(dataset_root, "json")

    # 创建输出目录结构
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dataset_root, folder), exist_ok=True)

    # 获取并划分数据集
    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))
    train_files, val_files = train_test_split(json_files, test_size=0.18, random_state=42)

    # 处理训练集
    for json_file in train_files:
        if not process_single_json(json_file,
                                   os.path.join(dataset_root, "labels/train"),
                                   os.path.join(dataset_root, "images/train")):
            delete_invalid_files(json_file)

    # 处理验证集
    for json_file in val_files:
        if not process_single_json(json_file,
                                   os.path.join(dataset_root, "labels/val"),
                                   os.path.join(dataset_root, "images/val")):
            delete_invalid_files(json_file)

    print("\n数据集转换完成！")
    print(f"有效训练样本: {len(train_files) - len([f for f in train_files if not os.path.exists(f)])}")
    print(f"有效验证样本: {len(val_files) - len([f for f in val_files if not os.path.exists(f)])}")