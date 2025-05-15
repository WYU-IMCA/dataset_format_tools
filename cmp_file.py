import os
import sys
#比较两个文件夹中的文件名，找出不匹配的文件

def find_mismatched_files(folder1, folder2):
    # 收集两个文件夹中的主名集合
    stems1 = set()
    stems2 = set()

    for filename in os.listdir(folder1):
        filepath = os.path.join(folder1, filename)
        if os.path.isfile(filepath):
            stem = os.path.splitext(filename)[0]
            stems1.add(stem)

    for filename in os.listdir(folder2):
        filepath = os.path.join(folder2, filename)
        if os.path.isfile(filepath):
            stem = os.path.splitext(filename)[0]
            stems2.add(stem)

    mismatched = []

    # 检查folder1中的文件
    for filename in os.listdir(folder1):
        filepath = os.path.join(folder1, filename)
        if os.path.isfile(filepath):
            stem = os.path.splitext(filename)[0]
            if stem not in stems2:
                mismatched.append(filepath)

    # 检查folder2中的文件
    for filename in os.listdir(folder2):
        filepath = os.path.join(folder2, filename)
        if os.path.isfile(filepath):
            stem = os.path.splitext(filename)[0]
            if stem not in stems1:
                mismatched.append(filepath)

    return mismatched

if __name__ == "__main__":

    folder1, folder2 = r"G:\yolov8\datasets\rune_all\json", r"G:\yolov8\datasets\rune_all\backup"

    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Error: Both arguments must be valid directories.")
        sys.exit(1)

    mismatched_files = find_mismatched_files(folder1, folder2)

    if mismatched_files:
        print("Mismatched files:")
        for file in mismatched_files:
            print(file)
    else:
        print("All files have matching counterparts.")