# import os
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import nibabel as nib           # 用于读取 nii.gz 格式的医学图像
# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# import glob
# import cv2
# import SimpleITK as sitk

# # 如果需要：
# import monai
# from monai.transforms import (
#     Compose,
#     SpatialPadd,
#     RandCropByPosNegLabeld,
#     CropForegroundd,
#     RandFlipd,
#     RandRotated,
#     RandZoomd,
#     RandAdjustContrastd,
#     RandShiftIntensityd,
#     MapTransform,
#     Lambda,
#     CenterSpatialCropd
# )

# # ---------------------------
# # 根据标签 nii.gz 计算 x-y 方向的全局 ROI bounding box
# # （如果你只想用RandCropByPosNegLabeld，可不再依赖此函数做2D裁剪）
# # ---------------------------
# def get_global_bounding_box_from_label(label_path):
#     label_img = nib.load(label_path)
#     label_data = label_img.get_fdata().astype(np.uint8)
#     # 转置为 (D, H, W)
#     label_data = np.transpose(label_data, (2, 0, 1))
#     coords = np.argwhere(label_data != 0)
#     if coords.size == 0:
#         D, H, W = label_data.shape
#         return [0, 0, W, H]
#     y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
#     y_max, x_max = coords[:, 1].max(), coords[:, 2].max()
#     return [int(x_min), int(y_min), int(x_max), int(y_max)]

# # ---------------------------
# # 根据标签 nii.gz 计算 z 轴上病灶区域范围
# # ---------------------------
# def get_roi_depth_range_from_label(label_path):
#     label_img = nib.load(label_path)
#     label_data = label_img.get_fdata().astype(np.uint8)
#     label_data = np.transpose(label_data, (2, 0, 1))
#     coords = np.argwhere(label_data != 0)
#     D = label_data.shape[0]
#     if coords.size == 0:
#         return (0, D - 1)
#     z_min = int(coords[:, 0].min())
#     z_max = int(coords[:, 0].max())
#     return (z_min, z_max)

# # ---------------------------
# # 辅助函数：固定体数据深度
# # ---------------------------
# def fix_depth(volume, fixed_depth=32, roi_depth_range=None):
#     """
#     volume: (C, D, H, W)
#     fixed_depth: 希望统一的深度大小
#     roi_depth_range: (z_min, z_max)，病灶深度范围
#     """
#     _, D, H, W = volume.shape
#     if D < fixed_depth:
#         pad_total = fixed_depth - D
#         pad_before = pad_total // 2
#         pad_after = pad_total - pad_before
#         volume = torch.nn.functional.pad(volume, (0, 0, 0, 0, pad_before, pad_after))
#         return volume
#     else:
#         if roi_depth_range is None:
#             start = (D - fixed_depth) // 2
#             volume = volume[:, start:start+fixed_depth, :, :]
#         else:
#             z_min, z_max = roi_depth_range
#             desired_center = (z_min + z_max) // 2
#             crop_start = desired_center - fixed_depth // 2
#             crop_start = max(0, min(crop_start, D - fixed_depth))
#             volume = volume[:, crop_start:crop_start+fixed_depth, :, :]
#         return volume

# class ForegroundCenterCropd(MapTransform):
#     """
#     根据标签计算前景区域的中心位置，然后以此为中心对 image 和 label 进行裁剪，
#     输出固定尺寸的 patch。如果前景区域不存在，则以图像中心为基准。
#     """
#     def __init__(self, keys, label_key, output_size, margin=0, allow_missing_keys=False):
#         super().__init__(keys, allow_missing_keys)
#         self.label_key = label_key
#         self.output_size = output_size  # 例如 (D, H, W)
#         self.margin = margin

#     def __call__(self, data):
#         d = dict(data)
#         label = d[self.label_key]
#         # 假设 label 的形状为 (C, D, H, W)，这里取第一个通道计算前景
#         label_np = label[0].cpu().numpy()
#         coords = np.argwhere(label_np > 0)
#         if coords.size == 0:
#             # 如果没有前景，使用图像中心
#             center = [s // 2 for s in label_np.shape]
#         else:
#             min_coords = coords.min(axis=0)
#             max_coords = coords.max(axis=0)
#             center = ((min_coords + max_coords) / 2).astype(int).tolist()

#         slices = []
#         for i, size in enumerate(self.output_size):
#             # 计算每个维度的裁剪起始位置
#             start = center[i] - size // 2 - self.margin
#             end = start + size
#             # 保证裁剪范围在有效区域内
#             start = max(start, 0)
#             if end > label_np.shape[i]:
#                 # 若超出图像边界，则调整起始位置
#                 start = max(label_np.shape[i] - size, 0)
#                 end = start + size
#             slices.append(slice(start, end))

#         for key in self.keys:
#             img = d[key]
#             # 假设 image/label 为 (C, D, H, W)，对除 channel 外的维度裁剪
#             d[key] = img[(slice(None),) + tuple(slices)]
#         return d

# # ---------------------------
# # 3D MRI 数据集类（已替换为 MONAI 版3D增强）
# # ---------------------------
# class MRIDataset3D(Dataset):
#     def __init__(self, image_paths, label_paths, diag_labels, fixed_depth=32, is_train=True):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.diag_labels = diag_labels
#         self.fixed_depth = fixed_depth
#         self.is_train = is_train

#         # 需要同时进行变换的键
#         self.keys = ["image", "label"]
#         self.angle = 15
#         self.final_shape = (35, 332, 332)

#         if self.is_train:
#             # 训练集3D数据增强流水线（与你提供的完全一致）
#             self.train_transforms = Compose([
#                 # CropForegroundd(
#                 #     keys=["image", "label"],
#                 #     source_key="label",           # 依据 label 的非零值区域来确定前景
#                 #     select_fn=lambda x: x > 0,    # 对于二值分割，前景通常是 > 0
#                 #     margin=(5, 30, 30),                     # 如果想在前景外再多留一些边缘，就把 margin 设大一些
#                 #     # spatial_size=self.final_shape,   # 需要的最终输出大小 (D, H, W)
#                 #     allow_smaller = False
#                 # ),
#                 SpatialPadd(
#                     keys=["image", "label"],
#                     spatial_size=self.final_shape,
#                     mode="constant",
#                     constant_values=0
#                 ),                
#                 # RandCropByPosNegLabeld(
#                 #     keys=["image", "label"],
#                 #     label_key="label",
#                 #     spatial_size=self.final_shape,
#                 #     pos=1,
#                 #     neg=0,
#                 #     num_samples=1
#                 # ),
#                 CenterSpatialCropd(
#                     keys=["image", "label"],
#                     roi_size=self.final_shape,
#                 ),
#                 RandFlipd(keys=self.keys, prob=0.5, spatial_axis=0),
#                 RandFlipd(keys=self.keys, prob=0.5, spatial_axis=1),
#                 RandFlipd(keys=self.keys, prob=0.5, spatial_axis=2),
#                 RandRotated(keys=self.keys, mode=("bilinear","nearest"),
#                             range_x=(-self.angle,self.angle), range_y=(-self.angle,self.angle), range_z=(-self.angle,self.angle)),
#                 RandZoomd(keys=self.keys, prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True, mode=("bilinear","nearest")),
#                 RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.5)),
#                 RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5)
#             ])
#         else:
#             # 测试集：可以只做最简单的变换（或你也可以保留一部分）
#             self.test_transforms = Compose([
#                 # # 使用自定义的确定性裁剪，以前景中心为基准
#                 # CropForegroundd(
#                 #     keys=["image", "label"],
#                 #     source_key="label",           # 依据 label 的非零值区域来确定前景
#                 #     select_fn=lambda x: x > 0,    # 对于二值分割，前景通常是 > 0
#                 #     margin=(5, 30, 30),                     # 如果想在前景外再多留一些边缘，就把 margin 设大一些
#                 #     # spatial_size=self.final_shape,   # 需要的最终输出大小 (D, H, W)
#                 #     allow_smaller = False
#                 # ),
#                 SpatialPadd(
#                     keys=["image", "label"],
#                     spatial_size=self.final_shape,
#                     mode="constant",
#                     constant_values=0
#                 ),
#                 CenterSpatialCropd(
#                     keys=["image", "label"],
#                     roi_size=self.final_shape,
#                 ),
#                 # monai.transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
#             ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         label_path = self.label_paths[idx]

#         # （1）读取原始 3D 图像/标签
#         img_obj = nib.load(image_path)
#         volume = img_obj.get_fdata().astype(np.float32)
#         # 转为 (D,H,W) -> 再加1个通道变成 (C=1, D, H, W)
#         volume_3d = np.expand_dims(np.transpose(volume, (2, 0, 1)), axis=0)

#         label_obj = nib.load(label_path)
#         label_data = label_obj.get_fdata().astype(np.float32)
#         label_3d = np.expand_dims(np.transpose(label_data, (2, 0, 1)), axis=0)
#         # 二值化
#         label_3d = (label_3d != 0).astype(np.float32)

#         # （2）转成 torch.Tensor，构造字典
#         sample_dict = {
#             "image": torch.tensor(volume_3d, dtype=torch.float32),
#             "label": torch.tensor(label_3d, dtype=torch.float32)
#         }

#         # （3）根据 is_train 决定使用哪条数据增强流水线
#         if self.is_train:
#             sample_dict = self.train_transforms(sample_dict)
#         else:
#             sample_dict = self.test_transforms(sample_dict)

#         # 提取增强后的图像和标签 (C, D, H, W)
#         aug_image = sample_dict["image"]
#         aug_label = sample_dict["label"]
#         # print(aug_image.shape, aug_label.shape)

#         # （4）组合成双通道：第一通道=原图, 第二通道=前景图
#         product_tensor = aug_image * aug_label
#         combined_tensor = torch.cat([aug_image, product_tensor], dim=0)
#         # 此时 combined_tensor 形状: (2, D, H, W)

#         # # （5）可选：依据病灶范围固定深度（若你仍想保留这一步）
#         # # 先算一下 label 的 ROI 范围
#         # z_min, z_max = get_roi_depth_range_from_label(label_path)
#         # combined_tensor = fix_depth(combined_tensor, self.fixed_depth, (z_min, z_max))

#         # （6）返回图像（2通道）+ 诊断标签
#         diag_label = self.diag_labels[idx]
#         return combined_tensor, diag_label
# # ---------------------------
# # 构建 DataLoader 示例
# # ---------------------------
# if __name__ == '__main__':
#     # 指定数据所在的基本路径
#     base_path = '/home/vipuser/Desktop/Data/Task02_PASp62'

#     # 获取训练图像与标签路径列表
#     train_image_paths = [os.path.join(base_path, 'imagesTr', file) 
#                           for file in os.listdir(os.path.join(base_path, 'imagesTr'))]
#     train_label_paths = [os.path.join(base_path, 'labelsTr', file) 
#                           for file in os.listdir(os.path.join(base_path, 'labelsTr'))]
    
#     # 根据标签文件判断诊断标签（0 或 1），构造诊断标签列表
#     train_diag_labels = []
#     for file in os.listdir(os.path.join(base_path, 'labelsTr')):
#         if glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
#            or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')):
#             train_diag_labels.append(0)
#         elif glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
#              or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file[:-len(".nii.gz")]+'*.nii.gz')):
#             train_diag_labels.append(1)
#         else:
#             raise RuntimeError()

#     idx = np.random.randint(len(train_label_paths))
#     # 为测试方便，这里直接指定两个样本的路径和标签
#     train_image_paths =  [train_image_paths[idx]] 
#     print(train_image_paths)

#     train_label_paths = [train_label_paths[idx]] 
#     train_diag_labels = [train_diag_labels[idx]] 

#     # 创建 MRIDataset2D 数据集实例，指定 is_train=True 表示使用训练时的增强流水线
#     train_dataset = MRIDataset3D(train_image_paths, train_label_paths, train_diag_labels, is_train=False, fixed_depth=32)

#     # 创建 DataLoader，此处 batch_size 设置为 25，并使用 4 个工作进程加载数据
#     train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=4)

#     # 检查一个批次数据的形状及标签
#     for images, labels in train_loader:
#         print("训练批次图像形状:", images.shape)
#         print("训练批次标签:", labels)
        
#         # ---------------------------
#         # 展示一个样本的中间切片
#         # 假设输出形状为 (batch, 2, D, 128, 128)
#         # ---------------------------
#         sample = images[0]  # 取第一个样本，形状为 (2, D, 128, 128)
#         D = sample.shape[1]
#         mid_slice = D // 2  # 取中间切片
#         # 分别提取两个通道的中间切片
#         image_channel = sample[0, mid_slice, :, :].cpu().numpy()
#         label_channel = sample[1, mid_slice, :, :].cpu().numpy()

#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(image_channel, cmap='gray')
#         plt.title("Image Channel")
#         plt.axis("off")
#         plt.subplot(1, 2, 2)
#         plt.imshow(label_channel, cmap='gray')
#         plt.title("Label Channel (Image*Mask)")
#         plt.axis("off")
#         plt.show()
#         # 将图像保存到当前工作目录
#         plt.savefig('output.png')
#         print("图像已保存为 output.png")
#         break




#     # import torch
#     # from monai.transforms import (
#     #     Compose,
#     #     SpatialPadd,
#     #     RandCropByPosNegLabeld,
#     #     RandFlipd,
#     #     RandRotate90d,
#     #     RandZoomd,
#     #     RandAdjustContrastd,
#     #     RandShiftIntensityd
#     # )

#     # # 定义需要同时进行变换的键
#     # keys = ["image", "label"]

#     # # 构建训练集数据增强流水线
#     # train_transforms = Compose([
#     #     SpatialPadd(keys=["image", "label"], spatial_size=(55, 280, 392), mode="constant", constant_values=0),
#     #     # 1. 随机裁剪（空间变换）：对 image 和 label 同时裁剪出大小为 (128,128,128) 的区域
#     #     RandCropByPosNegLabeld(
#     #         keys=["image", "label"],
#     #         label_key="label",           # 指定标签键，依据该标签判断 ROI
#     #         spatial_size=(44, 180, 248),  # 裁剪区域大小
#     #         pos=1,          # 50%的概率裁剪包含正样本的区域
#     #         neg = 19,
#     #         num_samples=1                # 每次生成一个样本
#     #     ),
        
#     #     # 2. 随机翻转：沿各个空间轴独立以 50% 的概率翻转（保证 image 和 label 同步翻转）
#     #     RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
#     #     RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
#     #     RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        
#     #     # 3. 随机旋转：这里采用 90° 的旋转，可选择连续旋转的 RandRotated
#     #     RandRotate90d(keys=keys, prob=0.5, max_k=3),
        
#     #     # 4. 随机缩放：以 90%～110% 的缩放因子进行缩放，同时保持尺寸一致
#     #     RandZoomd(keys=keys, prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True),
        
#     #     # 5. 随机颜色/强度变换：此部分只作用于图像，不改变分割 mask
#     #     RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.5)),
#     #     RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5)
#     # ])

#     # # 示例：假设 image_tensor 和 mask_tensor 均为 torch.Tensor
#     # # image_tensor 的形状可能是 (1, D, H, W)，mask_tensor 的形状也是 (1, D, H, W)
#     # sample = {
#     #     "image": torch.randn(1, 160, 160, 160),  # 模拟一个3D体数据（例如 MRI 图像）
#     #     "label": (torch.randn(1, 160, 160, 160) > 0).float()  # 模拟一个二值分割 mask
#     # }

#     # # 应用数据增强流水线
#     # augmented_sample = train_transforms(sample)

#     # # 将列表中的张量取出来
#     # image_tensor = augmented_sample[0]["image"]
#     # label_tensor = augmented_sample[0]["label"]

#     # # 查看结果形状
#     # print("Image shape:", image_tensor.shape)
#     # print("Label shape:", label_tensor.shape)




import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib           # 用于读取 nii.gz 格式的医学图像
import torchvision.transforms as T
import matplotlib.pyplot as plt
import glob
import cv2
import SimpleITK as sitk

# 如果需要：
import monai
from monai.transforms import (
    Compose,
    SpatialPadd,
    RandCropByPosNegLabeld,
    CropForegroundd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    MapTransform,
    Lambda,
    CenterSpatialCropd,
    Resized
)

# ---------------------------
# 根据标签 nii.gz 计算 x-y 方向的全局 ROI bounding box
# （如果你只想用RandCropByPosNegLabeld，可不再依赖此函数做2D裁剪）
# ---------------------------
def get_global_bounding_box_from_label(label_path):
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata().astype(np.uint8)
    # 转置为 (D, H, W)
    label_data = np.transpose(label_data, (2, 0, 1))
    coords = np.argwhere(label_data != 0)
    if coords.size == 0:
        D, H, W = label_data.shape
        return [0, 0, W, H]
    y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
    y_max, x_max = coords[:, 1].max(), coords[:, 2].max()
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

# ---------------------------
# 根据标签 nii.gz 计算 z 轴上病灶区域范围
# ---------------------------
def get_roi_depth_range_from_label(label_path):
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata().astype(np.uint8)
    label_data = np.transpose(label_data, (2, 0, 1))
    coords = np.argwhere(label_data != 0)
    D = label_data.shape[0]
    if coords.size == 0:
        return (0, D - 1)
    z_min = int(coords[:, 0].min())
    z_max = int(coords[:, 0].max())
    return (z_min, z_max)

# ---------------------------
# 辅助函数：固定体数据深度
# ---------------------------
def fix_depth(volume, fixed_depth=32, roi_depth_range=None):
    """
    volume: (C, D, H, W)
    fixed_depth: 希望统一的深度大小
    roi_depth_range: (z_min, z_max)，病灶深度范围
    """
    _, D, H, W = volume.shape
    if D < fixed_depth:
        pad_total = fixed_depth - D
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        volume = torch.nn.functional.pad(volume, (0, 0, 0, 0, pad_before, pad_after))
        return volume
    else:
        if roi_depth_range is None:
            start = (D - fixed_depth) // 2
            volume = volume[:, start:start+fixed_depth, :, :]
        else:
            z_min, z_max = roi_depth_range
            desired_center = (z_min + z_max) // 2
            crop_start = desired_center - fixed_depth // 2
            crop_start = max(0, min(crop_start, D - fixed_depth))
            volume = volume[:, crop_start:crop_start+fixed_depth, :, :]
        return volume

class ForegroundCenterCropd(MapTransform):
    """
    根据标签计算前景区域的中心位置，然后以此为中心对 image 和 label 进行裁剪，
    输出固定尺寸的 patch。如果前景区域不存在，则以图像中心为基准。
    """
    def __init__(self, keys, label_key, output_size, margin=0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.output_size = output_size  # 例如 (D, H, W)
        self.margin = margin

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        # 假设 label 的形状为 (C, D, H, W)，这里取第一个通道计算前景
        label_np = label[0].cpu().numpy()
        coords = np.argwhere(label_np > 0)
        if coords.size == 0:
            # 如果没有前景，使用图像中心
            center = [s // 2 for s in label_np.shape]
        else:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = ((min_coords + max_coords) / 2).astype(int).tolist()

        slices = []
        for i, size in enumerate(self.output_size):
            # 计算每个维度的裁剪起始位置
            start = center[i] - size // 2 - self.margin
            end = start + size
            # 保证裁剪范围在有效区域内
            start = max(start, 0)
            if end > label_np.shape[i]:
                # 若超出图像边界，则调整起始位置
                start = max(label_np.shape[i] - size, 0)
                end = start + size
            slices.append(slice(start, end))

        for key in self.keys:
            img = d[key]
            # 假设 image/label 为 (C, D, H, W)，对除 channel 外的维度裁剪
            d[key] = img[(slice(None),) + tuple(slices)]
        return d

# ---------------------------
# 3D MRI 数据集类（已替换为 MONAI 版3D增强）
# ---------------------------
class MRIDataset3D(Dataset):
    def __init__(self, image_paths, label_paths, diag_labels, fixed_depth=32, is_train=True, is_center3=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.diag_labels = diag_labels
        self.fixed_depth = fixed_depth
        self.is_train = is_train

        # 需要同时进行变换的键
        self.keys = ["image", "label"]
        self.angle = 15
        self.final_shape = (40, 250, 250)    # center1, center2
        if is_center3: self.final_shape = (45, 300, 300)    # center3
        
        if self.is_train:
            # 训练集3D数据增强流水线（与你提供的完全一致）
            self.train_transforms = Compose([
                # 直接替换原来的pad和crop操作
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.final_shape,  # 将原始尺寸(35, 332, 332)降低为一半
                    mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻插值
                    align_corners=(True, None) 
                ),
                RandFlipd(keys=self.keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=self.keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=self.keys, prob=0.5, spatial_axis=2),
                RandRotated(keys=self.keys, mode=("bilinear","nearest"),
                            range_x=(-self.angle,self.angle), range_y=(-self.angle,self.angle), range_z=(-self.angle,self.angle)),
                RandZoomd(keys=self.keys, prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True, mode=("bilinear","nearest")),
                RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.5)),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5)
            ])
        else:
            # 测试集：可以只做最简单的变换（或你也可以保留一部分）
            self.test_transforms = Compose([
                # # 使用自定义的确定性裁剪，以前景中心为基准
                # CropForegroundd(
                #     keys=["image", "label"],
                #     source_key="label",           # 依据 label 的非零值区域来确定前景
                #     select_fn=lambda x: x > 0,    # 对于二值分割，前景通常是 > 0
                #     margin=(5, 30, 30),                     # 如果想在前景外再多留一些边缘，就把 margin 设大一些
                #     # spatial_size=self.final_shape,   # 需要的最终输出大小 (D, H, W)
                #     allow_smaller = False
                # ),
                # 同样直接替换原来的pad和crop操作
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.final_shape,  # 将原始尺寸降低为一半
                    mode=("trilinear", "nearest"),
                    align_corners=(True, None) 
                ),
                # monai.transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # （1）读取原始 3D 图像/标签
        img_obj = nib.load(image_path)
        volume = img_obj.get_fdata().astype(np.float32)
        # 转为 (D,H,W) -> 再加1个通道变成 (C=1, D, H, W)
        volume_3d = np.expand_dims(np.transpose(volume, (2, 0, 1)), axis=0)

        label_obj = nib.load(label_path)
        label_data = label_obj.get_fdata().astype(np.float32)
        label_3d = np.expand_dims(np.transpose(label_data, (2, 0, 1)), axis=0)
        # 二值化
        label_3d = (label_3d != 0).astype(np.float32)

        # （2）转成 torch.Tensor，构造字典
        sample_dict = {
            "image": torch.tensor(volume_3d, dtype=torch.float32),
            "label": torch.tensor(label_3d, dtype=torch.float32)
        }

        # （3）根据 is_train 决定使用哪条数据增强流水线
        if self.is_train:
            sample_dict = self.train_transforms(sample_dict)
        else:
            sample_dict = self.test_transforms(sample_dict)

        # 提取增强后的图像和标签 (C, D, H, W)
        aug_image = sample_dict["image"]
        aug_label = sample_dict["label"]
        # print(aug_image.shape, aug_label.shape)

        # （4）组合成双通道：第一通道=原图, 第二通道=前景图
        product_tensor = aug_image * aug_label
        combined_tensor = torch.cat([aug_image, product_tensor], dim=0)
        # 此时 combined_tensor 形状: (2, D, H, W)

        # # （5）可选：依据病灶范围固定深度（若你仍想保留这一步）
        # # 先算一下 label 的 ROI 范围
        # z_min, z_max = get_roi_depth_range_from_label(label_path)
        # combined_tensor = fix_depth(combined_tensor, self.fixed_depth, (z_min, z_max))

        # （6）返回图像（2通道）+ 诊断标签
        diag_label = self.diag_labels[idx]
        return combined_tensor, diag_label
# ---------------------------
# 构建 DataLoader 示例
# ---------------------------
if __name__ == '__main__':
    # 指定数据所在的基本路径
    base_path = '/home/vipuser/Desktop/Data/Task02_PASp62'

    # 获取训练图像与标签路径列表
    train_image_paths = [os.path.join(base_path, 'imagesTr', file) 
                          for file in os.listdir(os.path.join(base_path, 'imagesTr'))]
    train_label_paths = [os.path.join(base_path, 'labelsTr', file) 
                          for file in os.listdir(os.path.join(base_path, 'labelsTr'))]
    
    # 根据标签文件判断诊断标签（0 或 1），构造诊断标签列表
    train_diag_labels = []
    for file in os.listdir(os.path.join(base_path, 'labelsTr')):
        if glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
           or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')):
            train_diag_labels.append(0)
        elif glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
             or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file[:-len(".nii.gz")]+'*.nii.gz')):
            train_diag_labels.append(1)
        else:
            raise RuntimeError()

    idx = np.random.randint(len(train_label_paths))
    # 为测试方便，这里直接指定两个样本的路径和标签
    train_image_paths =  [train_image_paths[idx]] 
    print(train_image_paths)

    train_label_paths = [train_label_paths[idx]] 
    train_diag_labels = [train_diag_labels[idx]] 

    # 创建 MRIDataset2D 数据集实例，指定 is_train=True 表示使用训练时的增强流水线
    train_dataset = MRIDataset3D(train_image_paths, train_label_paths, train_diag_labels, is_train=False, fixed_depth=32)

    # 创建 DataLoader，此处 batch_size 设置为 25，并使用 4 个工作进程加载数据
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=4)

    # 检查一个批次数据的形状及标签
    for images, labels in train_loader:
        print("训练批次图像形状:", images.shape)
        print("训练批次标签:", labels)
        
        # ---------------------------
        # 展示一个样本的中间切片
        # 假设输出形状为 (batch, 2, D, 128, 128)
        # ---------------------------
        sample = images[0]  # 取第一个样本，形状为 (2, D, 128, 128)
        D = sample.shape[1]
        mid_slice = D // 2  # 取中间切片
        # 分别提取两个通道的中间切片
        image_channel = sample[0, mid_slice, :, :].cpu().numpy()
        label_channel = sample[1, mid_slice, :, :].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_channel, cmap='gray')
        plt.title("Image Channel")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(label_channel, cmap='gray')
        plt.title("Label Channel (Image*Mask)")
        plt.axis("off")
        plt.show()
        # 将图像保存到当前工作目录
        plt.savefig('output.png')
        print("图像已保存为 output.png")
        break

