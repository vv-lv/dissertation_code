#!/usr/bin/env python3
import os
import nibabel as nib

folder_images = "/home/vipuser/Desktop/Data/Task02_PASp61/imagesTr"
folder_labels = "/home/vipuser/Desktop/Data/Task02_PASp61/labelsTr"

# 获取两个文件夹内的 .nii.gz 文件列表
files_images = [f for f in os.listdir(folder_images) if f.endswith(".nii.gz")]
files_labels = [f for f in os.listdir(folder_labels) if f.endswith(".nii.gz")]

# 转换为集合，方便后续求交集、差集
set_images = set(files_images)
set_labels = set(files_labels)

# 交集：同时存在于两个文件夹中的文件名
common_files = set_images.intersection(set_labels)

if not common_files:
    print("两个文件夹之间没有同名的 .nii.gz 文件。")
else:
    print("在两者都有的文件中，对比图像 shape：\n")
    for filename in sorted(common_files):
        path_img = os.path.join(folder_images, filename)
        path_lbl = os.path.join(folder_labels, filename)
        
        # 使用 nibabel 读取 .nii.gz 文件
        nib_img = nib.load(path_img)
        nib_lbl = nib.load(path_lbl)
        
        # 获取图像数据的 shape
        shape_img = nib_img.shape
        shape_lbl = nib_lbl.shape
        
        if shape_img == shape_lbl:
            print(f"[匹配] {filename} 的图像维度一致: {shape_img}")
        else:
            print(f"[不匹配] {filename} 的图像维度不一致: {shape_img} vs {shape_lbl}")

# 只在 imagesTs_3 文件夹中存在，但不在 labelsTs_3 文件夹中的文件
unique_images = set_images - set_labels
if unique_images:
    print("\n只在 imagesTs_3 文件夹中存在的 .nii.gz 文件：")
    for f in sorted(unique_images):
        print(f)

# 只在 labelsTs_3 文件夹中存在，但不在 imagesTs_3 文件夹中的文件
unique_labels = set_labels - set_images
if unique_labels:
    print("\n只在 labelsTs_3 文件夹中存在的 .nii.gz 文件：")
    for f in sorted(unique_labels):
        print(f)



