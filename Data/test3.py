# 检查一个文件夹下的.nii.gz文件形状是否一致

import os
import nibabel as nib

# 定义文件夹路径
folder_path = r'/home/vipuser/Desktop/nnUNet/nnUNet_preprocessed/Dataset061_PASp61/gt_segmentations'

# 获取文件夹中所有 .nii.gz 文件
nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

# 初始化一个变量以存储第一个影像的维度
def check_dimensions(folder_path, nii_files):
    reference_shape = None
    inconsistent_files = []

    for nii_file in nii_files:
        file_path = os.path.join(folder_path, nii_file)
        img = nib.load(file_path)
        img_data = img.get_fdata()
        current_shape = img_data.shape

        if reference_shape is None:
            # 设置第一个文件的形状为参考形状
            reference_shape = current_shape
        else:
            # 检查当前图像形状是否与参考形状一致
            if current_shape != reference_shape:
                inconsistent_files.append((nii_file, current_shape))

    # 打印结果
    if inconsistent_files:
        print("参考形状为:", reference_shape)
        print("以下文件与参考形状不一致:")
        for file, shape in inconsistent_files:
            print(f"{file}: {shape}")
    else:
        print("所有文件的形状一致，形状为:", reference_shape)

# 运行检查函数
check_dimensions(folder_path, nii_files)
