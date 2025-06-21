# 检查每个数据标签匹配对形状是否一致

import os
import shutil
import json
import random
import nibabel as nib
from collections import Counter

# 设置随机数种子，以确保结果可重复
random.seed(42)

# 定义数据的路径
base_path = ""
centers = ["center1", "center2"]
# centers = ["center1"]
label_dirs = {"center1": "center1_label", "center2": "center2_label"}


# 建立影像和标签的对应关系
def get_matching_files():
    image_files = []
    label_files = []
    for center in centers:
        center_path_PAS = os.path.join(base_path, center, 'PAS')
        center_path_NOPAS = os.path.join(base_path, center, 'NOPAS')
        label_path = os.path.join(base_path, label_dirs[center])
        image_files += [f for f in os.listdir(center_path_PAS) if f.endswith(".nii.gz")]
        image_files += [f for f in os.listdir(center_path_NOPAS) if f.endswith(".nii.gz")]
        label_files += [f for f in os.listdir(label_path) if f.endswith(".nii.gz")]

    # 去掉文件名中的后缀以便匹配
    image_names = {f.split('_')[0]: f for f in image_files}
    label_names = {f.split('.')[0]: f for f in label_files}
    # 找出有对应标签的影像文件
    matching_pairs = []
    for image_id, image_file in image_names.items():
        if image_id in label_names:
            matching_pairs.append((image_file, label_names[image_id]))

    return matching_pairs

# 检查每个匹配对的影像和标签是否具有相同的形状
def check_matching_pairs(matching_pairs):
    inconsistent_pairs = []
    for image_file, label_file in matching_pairs:
        # 自动判断影像文件所在的目录
        for center in centers:
            if os.path.exists(os.path.join(base_path, center, 'PAS', image_file)):
                image_path = os.path.join(base_path, center, 'PAS', image_file)
                break
            elif os.path.exists(os.path.join(base_path, center, 'NOPAS', image_file)):
                image_path = os.path.join(base_path, center, 'NOPAS', image_file)
                break
        else:
            continue

        label_path = os.path.join(base_path, label_dirs[center], label_file)

        # 加载影像和标签
        image = nib.load(image_path)
        label = nib.load(label_path)

        # 获取影像和标签的数据形状
        image_shape = image.get_fdata().shape
        label_shape = label.get_fdata().shape

        # 检查是否相等
        if image_shape != label_shape:
            inconsistent_pairs.append((image_file, label_file, image_shape, label_shape))

    # 打印结果
    if inconsistent_pairs:
        print("以下匹配对的影像和标签形状不一致:")
        for image_file, label_file, image_shape, label_shape in inconsistent_pairs:
            print(f"{image_file} 和 {label_file} 的形状分别为 {image_shape} 和 {label_shape}")
    else:
        print("所有匹配对的影像和标签形状一致。")

# 遍历每个中心的数据并建立匹配
def main():
    matching_pairs = get_matching_files()
    check_matching_pairs(matching_pairs)

if __name__ == "__main__":
    main()
