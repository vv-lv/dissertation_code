'''
用于对部分预处理后的训练集和内部测试集进行剩余预处理，获得全部预处理后的训练集和内部测试集。
具体来讲，包括裁剪、重采样到spacing相同、标准化。
'''

import numpy as np
from scipy.ndimage import binary_fill_holes
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import map_coordinates
import os
from tqdm import tqdm
from monai.transforms import (
    NormalizeIntensity,
    Spacing
)
from monai.data import MetaTensor
import torch

def modify_affine_no_rotation(
    old_affine: np.ndarray,
    old_spacing: tuple,  # (Sx, Sy, Sz)
    new_spacing: tuple,  # (Sx_new, Sy_new, Sz_new)
    bbox: list = None    # [ [x0,x1], [y0,y1], [z0,z1] ]
):
    """
    基于“无旋转/无斜切”的假设来更新affine：
      1) 先根据 bbox 计算平移量
      2) 再把 affine 的对角线替换成 new_spacing
    """
    new_affine = old_affine.copy()
    
    # === 1) 如果有 bbox，需要加上“裁剪偏移”的平移量 ===
    if bbox is not None:
        x0, y0, z0 = bbox[0][0], bbox[1][0], bbox[2][0]
        
        # 在 (X, Y, Z) = (dim0, dim1, dim2) 的前提下：
        # old_spacing = (Sx, Sy, Sz)
        shift_x = x0 * old_spacing[0]
        shift_y = y0 * old_spacing[1]
        shift_z = z0 * old_spacing[2]
        
        # 将新图像原点的世界坐标移动
        new_affine[0, 3] += shift_x
        new_affine[1, 3] += shift_y
        new_affine[2, 3] += shift_z
    
    # === 2) 更新仿射矩阵对角线，把 old_spacing 换成 new_spacing ===
    # old_affine[0,0] ≈ ±Sx, [1,1] ≈ ±Sy, [2,2] ≈ ±Sz
    # 只要没有旋转，对角线上除符号外就是 spacing
    for i in range(3):
        old_len = abs(new_affine[i, i])
        ratio = abs(new_spacing[i]) / old_len  # 计算要缩放的比例
        new_affine[i, i] *= ratio             # 等效替换 spacing
    return new_affine


# 步骤 1: 生成非零区域模板 (nonzero_mask)
def generate_nonzero_mask(data):
    nonzero_mask = data != 0  # 对于每个体素，值不为零的地方为True
    # nonzero_mask = binary_fill_holes(nonzero_mask)  # 填充孔洞
    return nonzero_mask

# 步骤 2: 计算裁剪区域的边界框 (bounding_box)
def get_bounding_box(nonzero_mask):
    """
    nonzero_mask.shape = (X, Y, Z)
    返回一个 bbox = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    x_idxs, y_idxs, z_idxs = np.where(nonzero_mask != 0)
    
    x_min, x_max = x_idxs.min(), x_idxs.max() + 1
    y_min, y_max = y_idxs.min(), y_idxs.max() + 1
    z_min, z_max = z_idxs.min(), z_idxs.max() + 1
    
    return [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

# 步骤 3: 根据bounding box裁剪图像
def crop_image(data, bbox):
    """
    data.shape = (X, Y, Z)
    bbox = [[x0,x1], [y0,y1], [z0,z1]]
    """
    x0, x1 = bbox[0]
    y0, y1 = bbox[1]
    z0, z1 = bbox[2]
    # 切片裁剪
    cropped = data[x0:x1, y0:y1, z0:z1]
    return cropped


# 步骤 4: 裁剪标签图像
def crop_label(seg, bbox):
    """同理，如果标签 seg.shape = (X, Y, Z)，同样裁剪"""
    x0, x1 = bbox[0]
    y0, y1 = bbox[1]
    z0, z1 = bbox[2]
    cropped_seg = seg[x0:x1, y0:y1, z0:z1]
    return cropped_seg


# 步骤 5: 判断是否使用 nonzero_mask 进行归一化
def should_use_nonzero_mask(data, cropped_data):
    # 计算裁剪前后的体积
    original_volume = np.prod(data.shape)  # 原始图像的体积
    cropped_volume = np.prod(cropped_data.shape)  # 裁剪后的图像体积

    # 如果裁剪后的体积小于原始体积的 3/4，说明图像尺寸减小超过了 1/4
    if cropped_volume < original_volume * 0.75:
        return True  # 需要仅对非零区域进行归一化
    else:
        return False  # 不需要，只对整个图像归一化

# 主程序：裁剪图像数据和标签
def preprocess_data(data, seg):
    # 生成非零区域模板
    nonzero_mask = generate_nonzero_mask(data)

    # 计算裁剪区域的边界框
    bbox = get_bounding_box(nonzero_mask)

    # 裁剪图像数据
    cropped_data = crop_image(data, bbox)

    # 裁剪标签数据
    cropped_seg = crop_label(seg, bbox)

    # 判断是否使用 nonzero_mask 进行归一化
    use_nonzero_mask = should_use_nonzero_mask(data, cropped_data)

    return cropped_data, cropped_seg, use_nonzero_mask, bbox




# # 测试代码 (假设你有三维医学影像数据集)
# # 生成示例数据 (数据：X, Y, Z)
# image_path = '/home/vipuser/Desktop/Data/Task02_PASp61/imagesTr/00058385.nii.gz'  # 修改为你的数据集路径
# label_path = '/home/vipuser/Desktop/Data/Task02_PASp61/labelsTr/00058385.nii.gz'  # 设置输出文件夹路径
# data = nib.load(image_path).get_fdata()
# seg = nib.load(label_path).get_fdata()

# # 获取仿射矩阵 (affine matrix)
# affine = nib.load(image_path).affine

# # 体素大小 (spacing)，从仿射矩阵的前三个元素中提取
# spacing = affine[:3, 0:3].diagonal()

# # 打印spacing
# print("Spacing (mm per voxel):", spacing)

# # 调用预处理函数
# cropped_data, cropped_seg, use_nonzero_mask  = preprocess_data(data, seg)

# print("裁剪前的数据形状:", data.shape)
# print("裁剪前的标签形状:", seg.shape)
# # 输出结果形状
# print("裁剪后的数据形状:", cropped_data.shape)
# print("裁剪后的标签形状:", cropped_seg.shape)
# print(use_nonzero_mask)




# 计算目标spacing
def calculate_target_spacing(spacings, shapes, target_spacing_percentile=50, anisotropy_threshold=3):
    # 计算spacing的中值
    target = np.percentile(np.vstack(spacings), target_spacing_percentile, axis=0)
    mid_shape = np.percentile(np.vstack(shapes), target_spacing_percentile, axis=0)
    
    # 判断是否存在各向异性
    worst_spacing_axis = np.argmax(target)  # 获取 spacing 最大的轴
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    
    # 判断是否存在各向异性
    has_aniso_spacing = target[worst_spacing_axis] > (anisotropy_threshold * min(other_spacings))
    
    # 在 nnUNet 中, 还要判断体素的尺寸 (has_aniso_voxels)
    target_size = np.array([spacing * shape for spacing, shape in zip(target, mid_shape)])  # 假设原始尺寸是(256, 256, 128)
    other_sizes = [target_size[i] for i in other_axes]
    has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)
    
    # 是否存在各向异性
    do_separate_z = has_aniso_spacing and has_aniso_voxels
    
    # 如果存在各向异性并且有体素差异
    if do_separate_z:
        spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
        
        # 计算该维度的 10% 分位点
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        
        # 确保该维度的 spacing 不超过其他维度
        if target_spacing_of_that_axis < min(other_spacings):
            target_spacing_of_that_axis = max(min(other_spacings), target_spacing_of_that_axis) + 1e-5
            
        target[worst_spacing_axis] = target_spacing_of_that_axis  # 更新目标 spacing
    
    # 返回目标spacing，是否分离Z轴以及最大spacing轴
    return target, do_separate_z, worst_spacing_axis


# 计算新的图像尺寸
def calculate_new_shape(original_spacing, target_spacing, shape):
    # print(original_spacing, target_spacing, shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)
    return new_shape




# # 执行重采样
# def resample_image(data, original_spacing, target_spacing, new_shape, do_separate_z=False, axis=None):
#     data = np.expand_dims(data, axis=0) 
#     if do_separate_z:
#         if axis == 0:
#             new_shape_2d = new_shape[1:]
#         elif axis == 1:
#             new_shape_2d = new_shape[[0, 2]]
#         else:
#             new_shape_2d = new_shape[:-1]
#         reshaped_final_data = []
#         for c in range(data.shape[0]):
#             reshaped_data = []
#             for slice_id in range(data.shape[axis]):
#                 if axis == 0:
#                     reshaped_data.append(resize(data[c, slice_id], new_shape_2d, order=3))
#                 elif axis == 1:
#                     reshaped_data.append(resize(data[c, :, slice_id], new_shape_2d, order=3))
#                 else:
#                     reshaped_data.append(resize(data[c, :, :, slice_id], new_shape_2d, order=3))

#         if data.shape[axis] != new_shape[axis]:
#             # The following few lines are blatantly copied and modified from sklearn's resize()
#             rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
#             reshaped_data = np.array(reshaped_data)
#             orig_rows, orig_cols, orig_dim = reshaped_data.shape

#             row_scale = float(orig_rows) / rows
#             col_scale = float(orig_cols) / cols
#             dim_scale = float(orig_dim) / dim

#             map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
#             map_rows = row_scale * (map_rows + 0.5) - 0.5
#             map_cols = col_scale * (map_cols + 0.5) - 0.5
#             map_dims = dim_scale * (map_dims + 0.5) - 0.5

#             coord_map = np.array([map_rows, map_cols, map_dims])
#             reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=3, cval=0,
#                                                                     mode='nearest')[None])
#     else:
#         reshaped = []
#         for c in range(data.shape[0]):
#             reshaped.append(resize(data[c], new_shape, order=3)[None])
#         reshaped_final_data = np.vstack(reshaped)

#     return reshaped_final_data[0]

# # 重采样标签，使用最近邻插值
# def resample_segmentation(seg, new_shape):
#     return resize(seg, new_shape, order=0)

def monai_resample_image_and_label(image_3d, label_3d, old_affine, target_spacing):
    """
    用 MONAI 的 Spacing transform 进行重采样
      - image_3d: (H, W, D) 原图 (numpy数组)
      - label_3d: (H, W, D) 标签 (numpy数组)
      - old_affine: 4x4 仿射矩阵 (numpy数组)
      - target_spacing: tuple, 目标 spacing，如 (1.0, 1.0, 1.0)

    返回:
      new_image_3d, new_label_3d, new_affine
    """
    import numpy as np
    from monai.transforms import Spacing
    from monai.data import MetaTensor

    # # 1) 先把 (H, W, D) -> (D, H, W)
    # image_3d = np.moveaxis(image_3d, -1, 0)  # (D, H, W)
    # label_3d = np.moveaxis(label_3d, -1, 0)  # (D, H, W)

    # 2) 再加 channel 维度 -> (1, D, H, W)
    image_4d = np.expand_dims(image_3d, axis=0)
    label_4d = np.expand_dims(label_3d, axis=0)

    # 3) 转成 MetaTensor
    image_mt = MetaTensor(image_4d, affine=old_affine, meta=None)
    label_mt = MetaTensor(label_4d, affine=old_affine, meta=None)

    # 4) 分别创建 Spacing transform (图像用双线性，标签用最近邻)
    spacing_image = Spacing(pixdim=target_spacing, mode="bilinear", recompute_affine=True)
    spacing_label = Spacing(pixdim=target_spacing, mode="nearest", recompute_affine=True)

    # 5) 分别重采样
    new_image_t = spacing_image(image_mt)
    new_label_t = spacing_label(label_mt)

    # 6) 取出新的 affine
    new_affine = new_image_t.affine

    # 7) 转回 numpy，并去掉通道维度
    new_image_4d = new_image_t.cpu().numpy()  # (1, D', H', W')
    new_label_4d = new_label_t.cpu().numpy()  # (1, D', H', W')

    new_image_3d = new_image_4d[0]  # (D', H', W')
    new_label_3d = new_label_4d[0]  # (D', H', W')
    # new_image_3d = np.moveaxis(new_image_3d, 0, -1)  # (H', W', D')
    # new_label_3d = np.moveaxis(new_label_3d, 0, -1)  # (H', W', D')

    return new_image_3d, new_label_3d, new_affine





def save_processed_image(output_dir, image_data, affine, subdir, filename):
    output_path = os.path.join(output_dir, subdir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_img = nib.Nifti1Image(image_data, affine)
    nib.save(processed_img, output_path)

def save_processed_segmentation(output_dir, seg_data, affine, subdir, filename):
    output_path = os.path.join(output_dir, subdir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_seg = nib.Nifti1Image(seg_data, affine)
    nib.save(processed_seg, output_path)

def load_nifti_get_array_and_affine(path):
    nif = nib.load(path)
    data = nif.get_fdata()
    affine = nif.affine
    return data, affine




# 示例：对数据进行重采样
def resample_dataset(data_dir, output_dir, target_spacing_percentile=50, anisotropy_threshold=3):
    # 创建输出目录
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    spacings = []
    shapes = []
    imageTr_paths = sorted([os.path.join(data_dir, 'imagesTr', fname) for fname in os.listdir(os.path.join(data_dir, 'imagesTr'))])
    imageTs_paths = sorted([os.path.join(data_dir, 'imagesTs', fname) for fname in os.listdir(os.path.join(data_dir, 'imagesTs'))])
    
    # 计算目标spacing
    for img_path in imageTr_paths:
        img = nib.load(img_path)
        spacing = img.header.get_zooms()[:3]  # 获取spacing信息
        spacings.append(spacing)
        shapes.append(img.shape)
    
    # 计算目标spacing和是否需要分离Z轴，以及最大spacing的轴
    target_spacing, do_separate_z, axis = calculate_target_spacing(spacings, shapes, target_spacing_percentile, anisotropy_threshold)

    images = []
    labels = []

    # 加载并重采样每张图片
    for img_path in tqdm(imageTr_paths[:]):
        # 1) 读取图像+标签
        img_data, img_affine = load_nifti_get_array_and_affine(img_path)
        seg_path = img_path.replace('imagesTr', 'labelsTr')
        seg_data, seg_affine = load_nifti_get_array_and_affine(seg_path)

        # 2) 裁剪
        cropped_img, cropped_seg, use_nonzero_mask, bbox = preprocess_data(img_data, seg_data)

        # # 先根据 bbox 修正一下 old_affine 的平移量
        # corrected_affine = modify_affine_no_rotation(
        #     old_affine=img_affine,
        #     old_spacing=img.header.get_zooms()[:3],  # 原spacing
        #     new_spacing=img.header.get_zooms()[:3],  # 这一步只是为了加平移, 还不改变 spacing
        #     bbox=bbox
        # )

        # 3) 用 MONAI Spacing 做重采样
        new_image_3d, new_label_3d, new_affine = monai_resample_image_and_label(
            cropped_img, cropped_seg, old_affine=img_affine,
            target_spacing=target_spacing
        )
        # # 处理标签（假设标签和影像文件相同，只是标签文件夹不同）
        # reshaped_seg = resample_segmentation(seg, new_shape)


        # new_affine = modify_affine_no_rotation(
        #                 old_affine=img_affine,
        #                 old_spacing=img_spacing,
        #                 new_spacing=target_spacing,
        #                 bbox=bbox
        #             )

        # if use_nonzero_mask:
        #     print(11111)
        #     mask = reshaped_seg >= 0 # 图像中非0区域
        #     reshaped_img_data[mask] = (reshaped_img_data[mask] - reshaped_img_data[mask].mean()) / reshaped_img_data[mask].std()
        # else:
            # mask = reshaped_img_data != 0
            # reshaped_img_data[mask] = (reshaped_img_data[mask] - reshaped_img_data[mask].mean()) / reshaped_img_data[mask].std()
       
        # 4) 归一化 (依然使用 MONAI 的 NormalizeIntensity nonzero=True)
        new_image_3d = np.expand_dims(new_image_3d, axis=0)  
        transform = NormalizeIntensity(nonzero=False)
        new_image_3d = transform(new_image_3d).astype(np.float32)
        new_image_3d = new_image_3d[0]

        # 5) 保存
        filename = os.path.basename(img_path)
        save_processed_image(output_dir, new_image_3d, new_affine, 'imagesTr', filename)

        # 注意：标签直接是 int/uint 类型的话就不必再做额外处理
        new_label_3d = np.round(new_label_3d).astype(np.uint8)
        save_processed_segmentation(output_dir, new_label_3d, new_affine, 'labelsTr', filename)
        # print(new_image_3d.shape, new_label_3d.shape)


    # 加载并重采样每张图片
    for img_path in tqdm(imageTs_paths[:]):
        img_data, img_affine = load_nifti_get_array_and_affine(img_path)
        seg_path = img_path.replace('imagesTs', 'labelsTs')
        seg_data, seg_affine = load_nifti_get_array_and_affine(seg_path)

        cropped_img, cropped_seg, use_nonzero_mask, bbox = preprocess_data(img_data, seg_data)

        # # 先根据 bbox 修正一下 old_affine 的平移量
        # corrected_affine = modify_affine_no_rotation(
        #     old_affine=img_affine,
        #     old_spacing=img.header.get_zooms()[:3],  # 原spacing
        #     new_spacing=img.header.get_zooms()[:3],  # 这一步只是为了加平移, 还不改变 spacing
        #     bbox=bbox
        # )

        new_image_3d, new_label_3d, new_affine = monai_resample_image_and_label(
            cropped_img, cropped_seg, old_affine=img_affine,
            target_spacing=target_spacing
        )

        new_image_3d = np.expand_dims(new_image_3d, axis=0)  
        transform = NormalizeIntensity(nonzero=False)
        new_image_3d = transform(new_image_3d).astype(np.float32)
        new_image_3d = new_image_3d[0]

        filename = os.path.basename(img_path)
        save_processed_image(output_dir, new_image_3d, new_affine, 'imagesTs', filename)

        new_label_3d = np.round(new_label_3d).astype(np.uint8)
        save_processed_segmentation(output_dir, new_label_3d, new_affine, 'labelsTs', filename)


# 示例运行：指定数据集目录
data_dir = "/home/vipuser/Desktop/Data/Task02_PASp61"  # 修改为你的数据集路径
output_dir = "/home/vipuser/Desktop/Data/Task02_PASp62"
resample_dataset(data_dir, output_dir)
