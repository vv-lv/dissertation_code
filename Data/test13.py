'''
可视化程序
'''

# # 可视化
# import nibabel as nib
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# def load_nii_file(file_path):
#     """加载 .nii.gz 文件并返回数据"""
#     nifti_image = nib.load(file_path)
#     return nifti_image.get_fdata()

# def visualize_multiple_nii_images(image_files_o, image_files_n, label_files_o, label_files_n, slice_idx=50):
#     """可视化多个 .nii.gz 图像及其对应的标注"""
#     num_images = len(image_files_o)

#     # 设置绘图的子图布局
#     fig, axes = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))

#     # 对每个图像和标注进行可视化
#     for i, (image_file_o, image_file_n, label_file_o, label_file_n) in enumerate(zip(image_files_o, image_files_n, label_files_o, label_files_n)):
#         image_data_o = load_nii_file(image_file_o)
#         label_data_o = load_nii_file(label_file_o)
#         image_data_n = load_nii_file(image_file_n)
#         label_data_n = load_nii_file(label_file_n)

#         # 提取切片
#         image_slice_o = image_data_o[:, :, slice_idx]
#         label_slice_o = label_data_o[:, :, slice_idx]
#         image_slice_n = image_data_n[:, :, slice_idx]
#         label_slice_n = label_data_n[:, :, slice_idx]

#         # 显示原图像
#         axes[i, 0].imshow(image_slice_o.T, cmap='gray')
#         axes[i, 0].set_title(f'Original Image {os.path.basename(image_file_o)}')
#         axes[i, 0].axis('off')

#         # 显示原图像标签，带透明度
#         axes[i, 1].imshow(label_slice_o.T, cmap='jet', alpha=0.5)
#         axes[i, 1].set_title(f'Label {os.path.basename(label_file_o)}')
#         axes[i, 1].axis('off')

#         # 显示修改后的图像
#         axes[i, 2].imshow(image_slice_n.T, cmap='gray')
#         axes[i, 2].set_title(f'Overlay Image {os.path.basename(image_file_n)}')
#         axes[i, 2].axis('off')

#         # 显示修改后的标签图像
#         axes[i, 3].imshow(label_slice_n.T, cmap='jet', alpha=0.5)
#         axes[i, 3].set_title(f'Label {os.path.basename(label_file_n)}')
#         axes[i, 3].axis('off')

#     plt.tight_layout()
#     plt.show()
#     plt.savefig('output.png')  # 保存到当前工作目录
#     print("图像已保存为 output.png")


# def main():
#     input_dir1 = '/home/vipuser/Desktop/Data/Task02_PASp61'  # 数据集路径
#     input_dir2 = '/home/vipuser/Desktop/Data/Task02_PASp61_local'  # 数据集路径
#     image_files_o = [
#         os.path.join(input_dir1, 'imagesTr', '159110.nii.gz'),
#         os.path.join(input_dir1, 'imagesTr', '374234zhang3.nii.gz')
#     ]
#     label_files_o = [
#         os.path.join(input_dir1, 'labelsTr', '159110.nii.gz'),
#         os.path.join(input_dir1, 'labelsTr', '374234zhang3.nii.gz')
#     ]
#     image_files_n = [
#         os.path.join(input_dir2, 'imagesTr', '159110.nii.gz'),
#         os.path.join(input_dir2, 'imagesTr', '374234zhang3.nii.gz')
#     ]
#     label_files_n = [
#         os.path.join(input_dir2, 'labelsTr', '159110.nii.gz'),
#         os.path.join(input_dir2, 'labelsTr', '374234zhang3.nii.gz')
#     ]

#     visualize_multiple_nii_images(image_files_o, image_files_n, label_files_o, label_files_n, slice_idx=15)

# if __name__ == "__main__":
#     main()




# # 可视化
# nii_file2 = nib.load("/home/vipuser/Desktop/nnUNet/nnUNet_results/Dataset061_PASp61/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0_unet/validation/00061108.nii.gz")
# image_data2 = nii_file2.get_fdata()  # 形状 (512, 512, 22)

# from scipy.ndimage import binary_dilation, binary_erosion

# # 提取边缘
# dilated = binary_dilation(image_data2, iterations=3)   # 膨胀
# eroded = binary_erosion(image_data2, iterations=3)    # 腐蚀
# edges = dilated ^ eroded          # 边缘是膨胀与腐蚀的异或

# image_data2 = edges.astype(int)

# # 可视化原始图像和概率图
# slice_index = image_data.shape[2] // 2  # 中间切片索引
# image_slice = image_data[:, :, slice_index]  # 从 .nii.gz 中提取中间切片
# image_slice2 = image_data2[:, :, slice_index]  # 从 .nii.gz 中提取中间切片

# image_slice = image_slice * image_slice2

# # 绘图
# plt.figure(figsize=(12, 6))

# # 原始图像
# plt.subplot(1, 2, 1)
# plt.imshow(image_slice.T, cmap="gray", origin="lower")
# plt.title("Original Image")
# plt.axis("off")

# # 概率图
# plt.subplot(1, 2, 2)
# plt.imshow(image_slice2.T, cmap="gray", origin="lower")
# plt.title("Probability Map (Class 1)")
# plt.axis("off")

# # 保存图像到文件
# plt.savefig('output.png')  # 保存到当前工作目录
# print("图像已保存为 output.png")

# plt.show()




# # 可视化
# import nibabel as nib
# import matplotlib.pyplot as plt
# import numpy as np

# def load_and_display_nii_image(file_path, slice_idx=50):
#     """加载 .nii.gz 图像并显示指定切片"""
#     # 加载图像
#     # nifti_image = nib.load(file_path)
#     # image_data = nifti_image.get_fdata()  # 获取图像数据
#     image_data = np.load(file_path)  # 获取图像数据
#     image_data = image_data[0]
#     print(image_data.shape)

#     # 提取指定的切片（默认是第50个切片）
#     image_slice = image_data[:, :, slice_idx]

#     # 显示图像切片
#     plt.imshow(image_slice.T, cmap='gray')
#     plt.title(f"Slice {slice_idx} from {file_path}")
#     plt.axis('off')
#     plt.show()
#     plt.savefig('output.png')  # 保存到当前工作目录
#     print("图像已保存为 output.png")

# def main():
#     # 指定 .nii.gz 文件的路径
#     # nii_file_path = '/home/vipuser/Desktop/nnUNet/nnUNet_preprocessed/Dataset061_PASp61/gt_segmentations/00058385.nii.gz'  # 替换为你自己的文件路径
#     nii_file_path = '/home/vipuser/Desktop/Data/Task02_PASp62/imagesTr/00058385.nii.gz'
#     # 调用函数显示图像
#     load_and_display_nii_image(nii_file_path, slice_idx=15)  # 可以更改切片索引

# if __name__ == "__main__":
#     main()




import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 读取 .nii.gz 图像文件
def load_nifti_image(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

# 可视化图像的切片（显示多个轴的切片在同一张图像中）
def visualize_image_slices_in_one_figure(img_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 创建 1 行 3 列的子图
    
    # x轴的中间切片
    print(img_data.shape, np.unique(img_data))
    slice_x = img_data[img_data.shape[0] // 2, :, :]
    axes[0].imshow(slice_x.T, cmap="gray", origin="lower")
    axes[0].set_title("X-axis Slice")
    axes[0].axis('off')  # 关闭坐标轴
    
    # y轴的中间切片
    slice_y = img_data[:, img_data.shape[1] // 2, :]
    axes[1].imshow(slice_y.T, cmap="gray", origin="lower")
    axes[1].set_title("Y-axis Slice")
    axes[1].axis('off')  # 关闭坐标轴
    
    # z轴的中间切片
    slice_z = img_data[:, :, img_data.shape[2] // 2]
    axes[2].imshow(slice_z.T, cmap="gray", origin="lower")
    axes[2].set_title("Z-axis Slice")
    axes[2].axis('off')  # 关闭坐标轴
    
    plt.tight_layout()  # 调整子图间距
    plt.show()
    plt.savefig('output.png')  # 保存到当前工作目录
    print("图像已保存为 output.png")

# 示例：读取并显示图像的三个轴的切片
def display_nifti_image(file_path):
    img_data = load_nifti_image(file_path)

    # 可视化 x、y、z 轴的切片
    visualize_image_slices_in_one_figure(img_data)

# 运行示例代码
file_path = '/home/vipuser/Desktop/nnUNet/nnUNet_raw/Dataset060_PASp1/imagesTr/00059211_0000.nii.gz'  # 修改为实际文件路径
display_nifti_image(file_path)




# # 假设 mask1 和 mask2 是二维 numpy 数组，值为 0 或 1
# img_data = load_nifti_image('/home/vipuser/Desktop/Data/Task02_PASp62/labelsTs/427684.nii.gz')
# idx = img_data.shape[2] // 2 + 0
# # '/home/vipuser/Desktop/Data/Task02_PASp61/labelsTs/361810.nii.gz'
# mask1 = load_nifti_image('/home/vipuser/Desktop/Data/Task02_PASp62/labelsTs/427684.nii.gz')[:, :, idx]
# mask2 = load_nifti_image('/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTs/427684.nii.gz')[:, :, idx]
# plt.figure(figsize=(6,6))
# plt.imshow(mask1, cmap='Reds', alpha=0.5)   # 第一张 mask，红色调，半透明
# plt.imshow(mask2, cmap='Blues', alpha=0.5)   # 第二张 mask，蓝色调，半透明
# plt.axis('off')
# plt.show()
# plt.savefig('output1.png')  # 保存到当前工作目录
# print("图像已保存为 output1.png")

