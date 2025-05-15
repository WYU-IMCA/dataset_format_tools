import os
import json

keypoint_is_visual=False

# 将目标检测框的信息转为yolo格式
def cc2yolo_bbox(img_width, img_height, bbox):
    dw = 1. / img_width
    dh = 1. / img_height
    # yolo中框的坐标信息为 框中心点的横纵坐标及其宽高
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    # 归一化坐标信息
    x = format(x * dw, '.5f')
    w = format(w * dw, '.5f')
    y = format(y * dh, '.5f')
    h = format(h * dh, '.5f')
    return (x, y, w, h)


# 将关键点keypoints的信息转为yolo格式
def cc2yolo_keypoints(img_width, img_height, keypoints):
    list = []
    dw = 1. / img_width
    dh = 1. / img_height
    keypoint_num = len(keypoints)
    for i in range(keypoint_num):
        # 每个关键点的横坐标数据
        if i % 3 == 0:
            list.append(format(keypoints[i] * dw, '.5f'))
        # 每个关键点的纵坐标数据
        if i % 3 == 1:
            list.append(format(keypoints[i] * dh, '.5f'))
        if i % 3 == 2:
            if not keypoint_is_visual:
                continue
            else:
                list.append(2)
    result = tuple(list)
    return result

def coco2txt(json_file_path,yolo_anno_path):
    # 指定COCO格式数据地址train
    data = json.load(open(json_file_path, 'r'))
    if not os.path.exists(yolo_anno_path):
        os.makedirs(yolo_anno_path)
    # 由于coco id是不连续的,会导致后面报错,所以这里生成一个map映射
    cate_id_map = {}
    num = 0
    for cate in data['categories']:
        cate_id_map[cate['id']] = num
        num += 1  # cate_id_map -> {87: 0, 1034: 1, 131: 2, 318: 3, 588: 4}

    for img in data['images']:
        # 获取图片文件名
        filename = img['file_name']
        # 图片宽度
        img_width = img['width']
        # 图片高度
        img_height = img['height']
        # 图片id
        img_id = img['id']
        # 生成的yolo格式标注的文件名
        yolo_txt_name = filename.split('.')[0] + '.txt'

        with open(yolo_anno_path +'/'+ yolo_txt_name, 'w') as f:
            # 遍历所有标注信息
            for anno in data['annotations']:
                # 若此标注中图片id等于所需的图片id
                if anno['image_id'] == img_id:
                    f.write(str(cate_id_map[anno['category_id']]) + ' ')
                    bbox_info = cc2yolo_bbox(img_width, img_height, anno['bbox'])
                    keypoints_info = cc2yolo_keypoints(img_width, img_height, anno['keypoints'])
                    for item in bbox_info:
                        f.write(item + ' ')
                    for item in keypoints_info:
                        f.write(str(item) + ' ')
                    f.write('\n')
        f.close()

if __name__ == '__main__':
    json_file_path = r'G:\yolov8\datasets\rune\annotations\keypoints_train.json'
    yolo_anno_path = r'G:\yolov8\datasets\rune'
    coco2txt(json_file_path, yolo_anno_path)