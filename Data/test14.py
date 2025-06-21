'''
用于对部分预处理后的训练集和内部测试集裁剪，获得集中于ROI的图像和标签。
'''

import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from scipy import ndimage


def get_label_edge_3d(label_array, morph_size_out, morph_size_in):
    """
    对 3D 标注做膨胀、腐蚀，然后求两者差(边缘)。
    label_array: (X, Y, Z)
    morph_size_out: 膨胀半径（体素单位）
    morph_size_in:  腐蚀半径（体素单位）
    返回值: 与 label_array 同维度顺序 (X, Y, Z) 的结果
    """
    # 1) 先转成 (Z, Y, X)
    label_transposed = np.transpose(label_array, (2, 1, 0)).astype(np.uint8)
    
    # 2) 交给 SimpleITK 做 3D 操作
    label_sitk = sitk.GetImageFromArray(label_transposed)
    eroded_sitk  = sitk.BinaryErode(label_sitk,  (morph_size_in, )*3)
    dilated_sitk = sitk.BinaryDilate(label_sitk, (morph_size_out,)*3)
    edge_sitk = sitk.Subtract(dilated_sitk, eroded_sitk)
    # label_sitk = sitk.GetImageFromArray(label_transposed)
    # eroded_sitk  = sitk.BinaryErode(label_sitk,  (morph_size_in, )*3)
    # dilated_sitk = sitk.BinaryDilate(eroded_sitk, (morph_size_out,)*3)
    # edge_sitk = dilated_sitk

    # 3) 拿回 Numpy，并转回 (X, Y, Z)
    edge_array_3d = sitk.GetArrayFromImage(edge_sitk)
    edge_array_3d = np.transpose(edge_array_3d, (2, 1, 0))

    return edge_array_3d

def get_label_edge_2d(label_array, morph_size_out, morph_size_in):
    """
    针对 3D 体数据的每个 z-slice (X, Y)，执行 2D 的膨胀、腐蚀，再求差。
    label_array: (X, Y, Z)
    morph_size_out / morph_size_in: 2D 膨胀、腐蚀的半径
    返回值: 与 label_array 同维度 (X, Y, Z) 的结果
    """
    # 准备一个空数组来存储结果
    edge_array_2d = np.zeros_like(label_array, dtype=np.uint8)
    # 在 Z 轴循环，对每一张 (X, Y) slice 做 2D 操作
    for z in range(label_array.shape[2]):
        slice_2d = label_array[..., z].astype(np.uint8)  # (X, Y)

        # 注意：对 2D 来说，SimpleITK 认为 shape 是 (row, col) => (Y, X)
        # 因此可以在这里先转置一下，但很多时候做 2D 操作不影响可视化次序
        # 如果想跟 SITK 的 (Y, X) 统一，可以这样：
        slice_2d_trans = np.transpose(slice_2d, (1,0))  # (Y, X)

        slice_sitk = sitk.GetImageFromArray(slice_2d_trans)
        dilated_2d = sitk.BinaryDilate(slice_sitk, (morph_size_out, morph_size_out))
        eroded_2d  = sitk.BinaryErode(slice_sitk,  (morph_size_in,  morph_size_in))
        edge_2d = sitk.Subtract(dilated_2d, eroded_2d)

        # 拿回 Numpy，并转置回 (X, Y)
        edge_2d_arr = sitk.GetArrayFromImage(edge_2d)
        edge_2d_arr = np.transpose(edge_2d_arr, (1,0))

        # 放回对应的 z-slice
        edge_array_2d[..., z] = edge_2d_arr

    return edge_array_2d

# def crop_image_with_dilation(image, label, expansion_size=2):
#     # label 形状 (X, Y, Z)
#     # 1) 先把 label 转置成 (Z, Y, X) 给 SITK
#     label_transposed = np.transpose(label, (2,1,0))  # (Z, Y, X)
#     label_sitk = sitk.GetImageFromArray(label_transposed.astype(np.uint8))

#     # 膨胀操作
#     dilated_label_sitk = sitk.BinaryDilate(label_sitk, (expansion_size,)*3)
#     expanded_label = sitk.GetArrayFromImage(dilated_label_sitk)  # 形状仍是 (Z, Y, X)

#     # 找到非零范围
#     z_idxs, y_idxs, x_idxs = np.nonzero(expanded_label)
#     z_min, z_max = np.min(z_idxs), np.max(z_idxs)
#     y_min, y_max = np.min(y_idxs), np.max(y_idxs)
#     x_min, x_max = np.min(x_idxs), np.max(x_idxs)

#     # 2) 对原图像 (X, Y, Z) 进行裁剪时需要注意顺序
#     # SITK 里的 z 对应原本数组的第三维
#     # 因此 cropped_image 在原图像要写成:
#     cropped_image = image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
#     # 同理裁剪label(也还没转回去的话，要先转回来……)

#     # 如果想保持返回的 cropped_label 也是 (X, Y, Z)，需要再转回来：
#     #   先从 expanded_label 里把感兴趣区域截取，然后再转置回来
#     cropped_label_sitk = label_transposed[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
#     cropped_label = np.transpose(cropped_label_sitk, (2,1,0))

#     return cropped_image, cropped_label


def process_images(input_dir, output_dir, image_subdir, label_subdir, expansion_size=4):
    # 遍历训练集和测试集的图像文件
    for image_file in tqdm(os.listdir(os.path.join(input_dir, image_subdir))):
        image_path = os.path.join(input_dir, image_subdir, image_file)
        label_path = os.path.join(input_dir, label_subdir, image_file)

        if image_file.endswith('.nii.gz') and os.path.exists(label_path):
            # 读取图像和标注
            image_nifti = nib.load(image_path)
            label_nifti = nib.load(label_path)
            # zoom = image_nifti.header.get_zooms()  # 例如返回 (1.2, 1.2, 6.0)
            # print("spacing =", zoom)

            image_data = image_nifti.get_fdata()
            label_data = label_nifti.get_fdata()

            # 截取原图像并扩展区域
            # cropped_image, cropped_label= crop_image_with_dilation(image_data, label_data, expansion_size)
            cropped_image, cropped_label= image_data, label_data
            
            # # 对裁剪后的标签提取边缘
            # cropped_label_2d = get_label_edge_2d(cropped_label, 8, 4)
            # cropped_label_3d = get_label_edge_3d(cropped_label_2d, 2, 0)
            # cropped_label = cropped_label_2d | cropped_label_3d
            # # cropped_label = cropped_label_3d




            # 计算内部和外部的距离变换
            dist_inside = ndimage.distance_transform_edt(label_data, sampling=(1.2, 1.2, 6.0))      # 前景到背景距离
            dist_outside = ndimage.distance_transform_edt(1 - label_data, sampling=(1.2, 1.2, 6.0))  # 背景到前景距离

            # 设定距离阈值t来控制边缘厚度（单位：体素）
            t_in = 6
            t_out = 8

            # 选取内部距边界<=t的点 (label_volume=1 表示前景内部)
            inner_band = (dist_inside <= t_in) & (cropped_label == 1)
            # 选取外部距边界<=t的点 (label_volume=0 表示物体外部)
            outer_band = (dist_outside <= t_out) & (cropped_label == 0)

            # 合并内部和外部边界带
            cropped_label = (inner_band | outer_band).astype(np.uint8)




            # 创建新的 NIfTI 图像并保存（图像保持不变，标签替换为边缘图）
            cropped_image_nifti = nib.Nifti1Image(cropped_image, image_nifti.affine)
            cropped_label_nifti = nib.Nifti1Image(cropped_label, label_nifti.affine)
            
            # 确保输出目录存在
            output_image_path = os.path.join(output_dir, image_subdir, image_file)
            output_label_path = os.path.join(output_dir, label_subdir, image_file)

            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

            # 保存到新目录
            nib.save(cropped_image_nifti, output_image_path)
            nib.save(cropped_label_nifti, output_label_path)

            # print(f"Processed {image_file} and saved to {output_image_path} and {output_label_path}")

def main():
    input_dir = '/home/vipuser/Desktop/Data/Task02_PASp62'  # 修改为你的数据集路径
    output_dir = '/home/vipuser/Desktop/Data/Task02_PASp62_edge'  # 设置输出文件夹路径
    expansion_size = 20  # 扩展区域的大小，可以根据需求调整

    # 处理训练集和测试集
    process_images(input_dir, output_dir, 'imagesTr', 'labelsTr', expansion_size)
    process_images(input_dir, output_dir, 'imagesTs', 'labelsTs', expansion_size)

if __name__ == "__main__":
    main()
    