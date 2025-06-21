import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    SpatialPadd,
    CenterSpatialCropd,
    CropForegroundd,
    Resized
)

class MRIDataset3DFusion(Dataset):
    """
    同时处理 global & local 图像 + label, 并返回 (global_2ch, local_2ch, diag_label).

    - global_img/global_label: 不做前景裁剪, 只做大尺寸Pad + CenterCrop
    - local_img/local_label : 首先执行 CropForegroundd(只对 local keys), 生成更小ROI,
      然后再与 global 一起做随机翻转/旋转/缩放/强度变换(保证二者保持同样的随机参数).
    """

    def __init__(
        self,
        global_image_paths,
        global_label_paths,
        local_image_paths,
        local_label_paths,
        diag_labels,
        is_train=True,
        is_center3=False
    ):
        super().__init__()
        self.global_image_paths = global_image_paths
        self.global_label_paths = global_label_paths
        self.local_image_paths  = local_image_paths
        self.local_label_paths  = local_label_paths
        self.diag_labels        = diag_labels
        self.is_train           = is_train
        self.final_shape_global = (40, 250, 250)    # center1, center2
        self.final_shape_local  = (37, 185, 250)    # center1, center2
        if is_center3:
            self.final_shape_global = (45, 300, 300)    # center3
            self.final_shape_local  = (42, 181, 252)    # center3

        # ------ 1) 只对 local 执行 Foreground Crop ------
        #     注意 keys 只写 image_local, label_local，
        #     source_key="label_local" 表示根据 label_local 的前景确定裁剪范围。
        self.crop_local_transform = Compose([
            CropForegroundd(
                keys=["image_local", "label_local"],
                source_key="label_local",
                select_fn=lambda x: x > 0,
                margin=(2, 20, 20),   # 这里相当于对前景外留一点边缘
                allow_smaller=False
            ),
        ])

        # ------ 2) 分别对 global/local 做 pad+centerCrop ------
        self.pad_crop_global = Compose([
            # 直接替换原来的pad和crop操作
            Resized(
                keys=["image_global", "label_global"],
                spatial_size=self.final_shape_global,  # 将原始尺寸(35, 332, 332)降低为一半
                mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻插值
                align_corners=(True, None) 
            ),
        ])

        self.pad_crop_local = Compose([
            SpatialPadd(
                keys=["image_local", "label_local"],
                spatial_size=self.final_shape_local,
                mode="constant",
                constant_values=0
            ),
            CenterSpatialCropd(
                keys=["image_local", "label_local"],
                roi_size=self.final_shape_local,
            ),
        ])

        # ------ 3) 公共随机增强: 同时对 global/local 做翻转/旋转/缩放/强度变换 ------
        #     注意 keys 中包含 ["image_global","label_global","image_local","label_local"]，以保证同步随机。
        self.common_augs = Compose([
            RandFlipd(
                keys=["image_global","label_global","image_local","label_local"],
                prob=0.5,
                spatial_axis=0
            ),
            RandFlipd(
                keys=["image_global","label_global","image_local","label_local"],
                prob=0.5,
                spatial_axis=1
            ),
            RandFlipd(
                keys=["image_global","label_global","image_local","label_local"],
                prob=0.5,
                spatial_axis=2
            ),
            RandRotated(
                keys=["image_global","label_global","image_local","label_local"],
                mode=("bilinear","nearest","bilinear","nearest"),
                range_x=(-15,15), range_y=(-15,15), range_z=(-15,15),
                prob=0.5,
            ),
            RandZoomd(
                keys=["image_global","label_global","image_local","label_local"],
                prob=0.5,
                min_zoom=0.9, max_zoom=1.1,
                keep_size=True,
                mode=("bilinear","nearest","bilinear","nearest")
            ),
            RandAdjustContrastd(
                keys=["image_global","image_local"],
                prob=0.5,
                gamma=(0.7, 1.5)
            ),
            RandShiftIntensityd(
                keys=["image_global","image_local"],
                offsets=0.1,
                prob=0.5
            )
        ])

    def __len__(self):
        return len(self.diag_labels)

    def __getitem__(self, idx):
        # ========== A) 读取 global 图像 + label ==========
        g_path = self.global_image_paths[idx]
        g_data = nib.load(g_path).get_fdata().astype(np.float32)
        g_data = np.expand_dims(np.transpose(g_data, (2,0,1)), axis=0)

        g_lbl_path = self.global_label_paths[idx]
        g_lbl_data = nib.load(g_lbl_path).get_fdata().astype(np.float32)
        g_lbl_data = (g_lbl_data != 0).astype(np.float32)
        g_lbl_data = np.expand_dims(np.transpose(g_lbl_data, (2,0,1)), axis=0)

        # ========== B) 读取 local 图像 + label ==========
        l_path = self.local_image_paths[idx]
        l_data = nib.load(l_path).get_fdata().astype(np.float32)
        l_data = np.expand_dims(np.transpose(l_data, (2,0,1)), axis=0)

        l_lbl_path = self.local_label_paths[idx]
        l_lbl_data = nib.load(l_lbl_path).get_fdata().astype(np.float32)
        l_lbl_data = (l_lbl_data != 0).astype(np.float32)
        l_lbl_data = np.expand_dims(np.transpose(l_lbl_data, (2,0,1)), axis=0)

        # ========== C) 构造一个 dict，用来套 MONAI 的 Compose ==========
        data_dict = {
            "image_global": torch.tensor(g_data, dtype=torch.float32),
            "label_global": torch.tensor(g_lbl_data, dtype=torch.float32),
            "image_local":  torch.tensor(l_data,  dtype=torch.float32),
            "label_local":  torch.tensor(l_lbl_data, dtype=torch.float32)
        }

        # ========== D) 只对 local 执行 CropForegroundd(第三阶段你只想裁剪局部) ==========
        data_dict = self.crop_local_transform(data_dict)

        # ========== E) 分别对 global/local 做 pad + center crop ==========
        data_dict = self.pad_crop_global(data_dict)  # 保证全局图尺寸 = final_shape_global
        data_dict = self.pad_crop_local(data_dict)   # 保证局部图尺寸 = final_shape_local

        # ========== F) 如果是训练，执行公共随机变换；测试则跳过 ==========
        if self.is_train:
            data_dict = self.common_augs(data_dict)

        # ========== G) 组装 2通道 (image, image*label) 并返回 ==========
        # global 分支
        g_img = data_dict["image_global"]  # shape: (1, Dg, Hg, Wg)
        g_lbl = data_dict["label_global"]
        g_2ch = torch.cat([g_img, g_img*g_lbl], dim=0)  # shape: (2, Dg, Hg, Wg)

        # local 分支
        l_img = data_dict["image_local"]
        l_lbl = data_dict["label_local"]
        l_2ch = torch.cat([l_img, l_img*l_lbl], dim=0)  # shape: (2, Dl, Hl, Wl)

        diag_label = self.diag_labels[idx]
        return g_2ch, l_2ch, diag_label


if __name__ == "__main__":
    import os
    import random
    from monai.transforms import Compose
    from torch.utils.data import DataLoader
    import glob
    import matplotlib.pyplot as plt

    # 指定数据所在的基本路径
    base_path_global = '/home/vipuser/Desktop/Data/Task02_PASp62'

    # 获取训练图像与标签路径列表
    global_image_paths = [os.path.join(base_path_global, 'imagesTr', file) 
                          for file in os.listdir(os.path.join(base_path_global, 'imagesTr'))]
    global_label_paths = [os.path.join(base_path_global, 'labelsTr', file) 
                          for file in os.listdir(os.path.join(base_path_global, 'labelsTr'))]
    
    # 根据标签文件判断诊断标签（0 或 1），构造诊断标签列表
    diag_labels = []
    for file in os.listdir(os.path.join(base_path_global, 'labelsTr')):
        if glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
           or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')):
            diag_labels.append(0)
        elif glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
             or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file[:-len(".nii.gz")]+'*.nii.gz')):
            diag_labels.append(1)
        else:
            raise RuntimeError()
    
        # 指定数据所在的基本路径
    base_path_edge = '/home/vipuser/Desktop/Data/Task02_PASp62_edge'

    # 获取训练图像与标签路径列表
    local_image_paths = [os.path.join(base_path_edge, 'imagesTr', file) 
                          for file in os.listdir(os.path.join(base_path_edge, 'imagesTr'))]
    local_label_paths = [os.path.join(base_path_edge, 'labelsTr', file) 
                          for file in os.listdir(os.path.join(base_path_edge, 'labelsTr'))]
    
    # 根据标签文件判断诊断标签（0 或 1），构造诊断标签列表
    local_diag_labels = []
    for file in os.listdir(os.path.join(base_path_edge, 'labelsTr')):
        if glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
           or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')):
            local_diag_labels.append(0)
        elif glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
             or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file[:-len(".nii.gz")]+'*.nii.gz')):
            local_diag_labels.append(1)
        else:
            raise RuntimeError()
        
    if np.unique(diag_labels == local_diag_labels) != True:
        print("两个数据集的病例没有完全匹配")
        raise RuntimeError()
    

    idx = np.random.randint(len(global_image_paths))
    # 为测试方便，这里直接指定两个样本的路径和标签
    global_image_paths =  [global_image_paths[idx]] 
    print(global_image_paths)

    global_label_paths = [global_label_paths[idx]] 
    diag_labels = [diag_labels[idx]] 
    # 为测试方便，这里直接指定两个样本的路径和标签
    local_image_paths =  [local_image_paths[idx]] 
    print(local_image_paths)

    local_label_paths = [local_label_paths[idx]] 
    local_diag_labels = [local_diag_labels[idx]] 


    # 创建融合数据集
    fusion_dataset = MRIDataset3DFusion(
        global_image_paths=global_image_paths,
        global_label_paths=global_label_paths,
        local_image_paths=local_image_paths,
        local_label_paths=local_label_paths,
        diag_labels=diag_labels,
        is_train=True  # 测试训练模式
    )

    # 创建 DataLoader
    fusion_loader = DataLoader(fusion_dataset, batch_size=2, shuffle=False)

    for batch_idx, (g_vol, l_vol, diag_labels) in enumerate(fusion_loader):
        # g_vol shape: (B, 2, Dg, Hg, Wg)
        # l_vol shape: (B, 2, Dl, Hl, Wl)
        # diag_labels: (B,)

        print(f"[Batch {batch_idx}]")
        print("Global Volume shape:", g_vol.shape)
        print("Local Volume shape :", l_vol.shape)
        print("Labels:", diag_labels)

        # 只演示第一个样本
        g_sample = g_vol[0]  # shape = (2, Dg, Hg, Wg)
        l_sample = l_vol[0]  # shape = (2, Dl, Hl, Wl)

        # 分别找全局图像和局部图像的 D 维中间切片
        Dg = g_sample.shape[1]
        Dl = l_sample.shape[1]
        mid_g = Dg // 2
        mid_l = Dl // 2

        # ---------- 全局图像可视化 ----------
        # global_image_channel     = g_sample[0, mid_g, :, :].cpu().numpy()
        # global_labelmask_channel= g_sample[1, mid_g, :, :].cpu().numpy()

        # ---------- 局部图像可视化 ----------
        # local_image_channel      = l_sample[0, mid_l, :, :].cpu().numpy()
        # local_labelmask_channel = l_sample[1, mid_l, :, :].cpu().numpy()

        global_image_channel      = g_sample[0, mid_g].cpu().numpy()
        global_labelmask_channel  = g_sample[1, mid_g].cpu().numpy()
        local_image_channel       = l_sample[0, mid_l].cpu().numpy()
        local_labelmask_channel   = l_sample[1, mid_l].cpu().numpy()

        # 画布设置：2行×2列，展示全局图像(上)，局部图像(下)
        plt.figure(figsize=(10, 8))

        # (1) Global - 原图通道
        plt.subplot(2, 2, 1)
        plt.imshow(global_image_channel, cmap='gray')
        plt.title("Global - Image Channel")
        plt.axis("off")

        # (2) Global - (图像×mask)通道
        plt.subplot(2, 2, 2)
        plt.imshow(global_labelmask_channel, cmap='gray')
        plt.title("Global - (Image*Mask) Channel")
        plt.axis("off")

        # (3) Local - 原图通道
        plt.subplot(2, 2, 3)
        plt.imshow(local_image_channel, cmap='gray')
        plt.title("Local - Image Channel")
        plt.axis("off")

        # (4) Local - (图像×mask)通道
        plt.subplot(2, 2, 4)
        plt.imshow(local_labelmask_channel, cmap='gray')
        plt.title("Local - (Image*Mask) Channel")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # 或者保存为文件
        plt.savefig('output.png')
        print("可视化结果已保存为 'output.png'")

        # 仅展示第一个 batch，这里 break
        break








# import nibabel as nib
# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# from monai.transforms import (
#     Compose,
#     RandFlipd,
#     RandRotated,
#     RandZoomd,
#     RandShiftIntensityd,
#     RandAdjustContrastd,
#     SpatialPadd,
#     CenterSpatialCropd,
#     CropForegroundd,
#     Resized
# )

# class MRIDataset3DFusion(Dataset):
#     """
#     同时处理 global & local 图像 + label, 并返回 (global_2ch, local_2ch, diag_label).

#     - global_img/global_label: 不做前景裁剪, 只做大尺寸Pad + CenterCrop
#     - local_img/local_label : 首先执行 CropForegroundd(只对 local keys), 生成更小ROI,
#       然后再与 global 一起做随机翻转/旋转/缩放/强度变换(保证二者保持同样的随机参数).
#     """

#     def __init__(
#         self,
#         global_image_paths,
#         global_label_paths,
#         local_image_paths,
#         local_label_paths,
#         diag_labels,
#         is_train=True,
#         is_center3=False
#     ):
#         super().__init__()
#         self.global_image_paths = global_image_paths
#         self.global_label_paths = global_label_paths
#         self.local_image_paths  = local_image_paths
#         self.local_label_paths  = local_label_paths
#         self.diag_labels        = diag_labels
#         self.is_train           = is_train
#         self.final_shape_global = (40, 250, 250)    # center1, center2
#         self.final_shape_local  = (37, 185, 250)    # center1, center2
#         if is_center3:
#             self.final_shape_global = (45, 300, 300)    # center3
#             self.final_shape_local  = (42, 181, 252)    # center3


#         # ------ 1) 只对 local 执行 Foreground Crop ------
#         #     注意 keys 只写 image_local, label_local，
#         #     source_key="label_local" 表示根据 label_local 的前景确定裁剪范围。
#         self.crop_local_transform = Compose([
#             CropForegroundd(
#                 keys=["image_local", "label_local"],
#                 source_key="label_local",
#                 select_fn=lambda x: x > 0,
#                 margin=(2, 20, 20),   # 这里相当于对前景外留一点边缘
#                 allow_smaller=False
#             ),
#         ])

#         # ------ 2) 分别对 global/local 做 pad+centerCrop ------
#         self.pad_crop_global = Compose([
#             # 直接替换原来的pad和crop操作
#             Resized(
#                 keys=["image_global", "label_global"],
#                 spatial_size=self.final_shape_global,  # 将原始尺寸(35, 332, 332)降低为一半
#                 mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻插值
#                 align_corners=(True, None) 
#             ),
#         ])

#         self.pad_crop_local = Compose([
#             # 直接替换原来的pad和crop操作
#             Resized(
#                 keys=["image_local", "label_local"],
#                 spatial_size=self.final_shape_local,  # 将原始尺寸(35, 332, 332)降低为一半
#                 mode=("trilinear", "nearest"),  # 图像用三线性插值，标签用最近邻插值
#                 align_corners=(True, None) 
#             ),
#         ])

#         # ------ 3) 公共随机增强: 同时对 global/local 做翻转/旋转/缩放/强度变换 ------
#         #     注意 keys 中包含 ["image_global","label_global","image_local","label_local"]，以保证同步随机。
#         self.common_augs = Compose([
#             RandFlipd(
#                 keys=["image_global","label_global","image_local","label_local"],
#                 prob=0.5,
#                 spatial_axis=0
#             ),
#             RandFlipd(
#                 keys=["image_global","label_global","image_local","label_local"],
#                 prob=0.5,
#                 spatial_axis=1
#             ),
#             RandFlipd(
#                 keys=["image_global","label_global","image_local","label_local"],
#                 prob=0.5,
#                 spatial_axis=2
#             ),
#             RandRotated(
#                 keys=["image_global","label_global","image_local","label_local"],
#                 mode=("bilinear","nearest","bilinear","nearest"),
#                 range_x=(-15,15), range_y=(-15,15), range_z=(-15,15),
#                 prob=0.5,
#             ),
#             RandZoomd(
#                 keys=["image_global","label_global","image_local","label_local"],
#                 prob=0.5,
#                 min_zoom=0.9, max_zoom=1.1,
#                 keep_size=True,
#                 mode=("bilinear","nearest","bilinear","nearest")
#             ),
#             RandAdjustContrastd(
#                 keys=["image_global","image_local"],
#                 prob=0.5,
#                 gamma=(0.7, 1.5)
#             ),
#             RandShiftIntensityd(
#                 keys=["image_global","image_local"],
#                 offsets=0.1,
#                 prob=0.5
#             )
#         ])

#     def __len__(self):
#         return len(self.diag_labels)

#     def __getitem__(self, idx):
#         # ========== A) 读取 global 图像 + label ==========
#         g_path = self.global_image_paths[idx]
#         g_data = nib.load(g_path).get_fdata().astype(np.float32)
#         g_data = np.expand_dims(np.transpose(g_data, (2,0,1)), axis=0)

#         g_lbl_path = self.global_label_paths[idx]
#         g_lbl_data = nib.load(g_lbl_path).get_fdata().astype(np.float32)
#         g_lbl_data = (g_lbl_data != 0).astype(np.float32)
#         g_lbl_data = np.expand_dims(np.transpose(g_lbl_data, (2,0,1)), axis=0)

#         # ========== B) 读取 local 图像 + label ==========
#         l_path = self.local_image_paths[idx]
#         l_data = nib.load(l_path).get_fdata().astype(np.float32)
#         l_data = np.expand_dims(np.transpose(l_data, (2,0,1)), axis=0)

#         l_lbl_path = self.local_label_paths[idx]
#         l_lbl_data = nib.load(l_lbl_path).get_fdata().astype(np.float32)
#         l_lbl_data = (l_lbl_data != 0).astype(np.float32)
#         l_lbl_data = np.expand_dims(np.transpose(l_lbl_data, (2,0,1)), axis=0)

#         # ========== C) 构造一个 dict，用来套 MONAI 的 Compose ==========
#         data_dict = {
#             "image_global": torch.tensor(g_data, dtype=torch.float32),
#             "label_global": torch.tensor(g_lbl_data, dtype=torch.float32),
#             "image_local":  torch.tensor(l_data,  dtype=torch.float32),
#             "label_local":  torch.tensor(l_lbl_data, dtype=torch.float32)
#         }

#         # ========== D) 只对 local 执行 CropForegroundd(第三阶段你只想裁剪局部) ==========
#         data_dict = self.crop_local_transform(data_dict)

#         # ========== E) 分别对 global/local 做 pad + center crop ==========
#         data_dict = self.pad_crop_global(data_dict)  # 保证全局图尺寸 = final_shape_global
#         data_dict = self.pad_crop_local(data_dict)   # 保证局部图尺寸 = final_shape_local

#         # ========== F) 如果是训练，执行公共随机变换；测试则跳过 ==========
#         if self.is_train:
#             data_dict = self.common_augs(data_dict)

#         # ========== G) 组装 2通道 (image, image*label) 并返回 ==========
#         # global 分支
#         g_img = data_dict["image_global"]  # shape: (1, Dg, Hg, Wg)
#         g_lbl = data_dict["label_global"]
#         g_2ch = torch.cat([g_img, g_img*g_lbl], dim=0)  # shape: (2, Dg, Hg, Wg)

#         # local 分支
#         l_img = data_dict["image_local"]
#         l_lbl = data_dict["label_local"]
#         l_2ch = torch.cat([l_img, l_img*l_lbl], dim=0)  # shape: (2, Dl, Hl, Wl)

#         diag_label = self.diag_labels[idx]
#         return g_2ch, l_2ch, diag_label


# if __name__ == "__main__":
#     import os
#     import random
#     from monai.transforms import Compose
#     from torch.utils.data import DataLoader
#     import glob
#     import matplotlib.pyplot as plt

#     # 指定数据所在的基本路径
#     base_path_global = '/home/vipuser/Desktop/Data/Task02_PASp62'

#     # 获取训练图像与标签路径列表
#     global_image_paths = [os.path.join(base_path_global, 'imagesTr', file) 
#                           for file in os.listdir(os.path.join(base_path_global, 'imagesTr'))]
#     global_label_paths = [os.path.join(base_path_global, 'labelsTr', file) 
#                           for file in os.listdir(os.path.join(base_path_global, 'labelsTr'))]
    
#     # 根据标签文件判断诊断标签（0 或 1），构造诊断标签列表
#     diag_labels = []
#     for file in os.listdir(os.path.join(base_path_global, 'labelsTr')):
#         if glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
#            or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')):
#             diag_labels.append(0)
#         elif glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
#              or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file[:-len(".nii.gz")]+'*.nii.gz')):
#             diag_labels.append(1)
#         else:
#             raise RuntimeError()
    
#         # 指定数据所在的基本路径
#     base_path_edge = '/home/vipuser/Desktop/Data/Task02_PASp62_edge'

#     # 获取训练图像与标签路径列表
#     local_image_paths = [os.path.join(base_path_edge, 'imagesTr', file) 
#                           for file in os.listdir(os.path.join(base_path_edge, 'imagesTr'))]
#     local_label_paths = [os.path.join(base_path_edge, 'labelsTr', file) 
#                           for file in os.listdir(os.path.join(base_path_edge, 'labelsTr'))]
    
#     # 根据标签文件判断诊断标签（0 或 1），构造诊断标签列表
#     local_diag_labels = []
#     for file in os.listdir(os.path.join(base_path_edge, 'labelsTr')):
#         if glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
#            or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file[:-len(".nii.gz")]+'*.nii.gz')):
#             local_diag_labels.append(0)
#         elif glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file[:-len(".nii.gz")]+'*.nii.gz')) \
#              or glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file[:-len(".nii.gz")]+'*.nii.gz')):
#             local_diag_labels.append(1)
#         else:
#             raise RuntimeError()
        
#     if np.unique(diag_labels == local_diag_labels) != True:
#         print("两个数据集的病例没有完全匹配")
#         raise RuntimeError()
    

#     idx = np.random.randint(len(global_image_paths))
#     # 为测试方便，这里直接指定两个样本的路径和标签
#     global_image_paths =  [global_image_paths[idx]] 
#     print(global_image_paths)

#     global_label_paths = [global_label_paths[idx]] 
#     diag_labels = [diag_labels[idx]] 
#     # 为测试方便，这里直接指定两个样本的路径和标签
#     local_image_paths =  [local_image_paths[idx]] 
#     print(local_image_paths)

#     local_label_paths = [local_label_paths[idx]] 
#     local_diag_labels = [local_diag_labels[idx]] 


#     # 创建融合数据集
#     fusion_dataset = MRIDataset3DFusion(
#         global_image_paths=global_image_paths,
#         global_label_paths=global_label_paths,
#         local_image_paths=local_image_paths,
#         local_label_paths=local_label_paths,
#         diag_labels=diag_labels,
#         is_train=True  # 测试训练模式
#     )

#     # 创建 DataLoader
#     fusion_loader = DataLoader(fusion_dataset, batch_size=2, shuffle=False)

#     for batch_idx, (g_vol, l_vol, diag_labels) in enumerate(fusion_loader):
#         # g_vol shape: (B, 2, Dg, Hg, Wg)
#         # l_vol shape: (B, 2, Dl, Hl, Wl)
#         # diag_labels: (B,)

#         print(f"[Batch {batch_idx}]")
#         print("Global Volume shape:", g_vol.shape)
#         print("Local Volume shape :", l_vol.shape)
#         print("Labels:", diag_labels)

#         # 只演示第一个样本
#         g_sample = g_vol[0]  # shape = (2, Dg, Hg, Wg)
#         l_sample = l_vol[0]  # shape = (2, Dl, Hl, Wl)

#         # 分别找全局图像和局部图像的 D 维中间切片
#         Dg = g_sample.shape[1]
#         Dl = l_sample.shape[1]
#         mid_g = Dg // 2
#         mid_l = Dl // 2

#         # ---------- 全局图像可视化 ----------
#         # global_image_channel     = g_sample[0, mid_g, :, :].cpu().numpy()
#         # global_labelmask_channel= g_sample[1, mid_g, :, :].cpu().numpy()

#         # ---------- 局部图像可视化 ----------
#         # local_image_channel      = l_sample[0, mid_l, :, :].cpu().numpy()
#         # local_labelmask_channel = l_sample[1, mid_l, :, :].cpu().numpy()

#         global_image_channel      = g_sample[0, mid_g].cpu().numpy()
#         global_labelmask_channel  = g_sample[1, mid_g].cpu().numpy()
#         local_image_channel       = l_sample[0, mid_l].cpu().numpy()
#         local_labelmask_channel   = l_sample[1, mid_l].cpu().numpy()

#         # 画布设置：2行×2列，展示全局图像(上)，局部图像(下)
#         plt.figure(figsize=(10, 8))

#         # (1) Global - 原图通道
#         plt.subplot(2, 2, 1)
#         plt.imshow(global_image_channel, cmap='gray')
#         plt.title("Global - Image Channel")
#         plt.axis("off")

#         # (2) Global - (图像×mask)通道
#         plt.subplot(2, 2, 2)
#         plt.imshow(global_labelmask_channel, cmap='gray')
#         plt.title("Global - (Image*Mask) Channel")
#         plt.axis("off")

#         # (3) Local - 原图通道
#         plt.subplot(2, 2, 3)
#         plt.imshow(local_image_channel, cmap='gray')
#         plt.title("Local - Image Channel")
#         plt.axis("off")

#         # (4) Local - (图像×mask)通道
#         plt.subplot(2, 2, 4)
#         plt.imshow(local_labelmask_channel, cmap='gray')
#         plt.title("Local - (Image*Mask) Channel")
#         plt.axis("off")

#         plt.tight_layout()
#         plt.show()

#         # 或者保存为文件
#         plt.savefig('output.png')
#         print("可视化结果已保存为 'output.png'")

#         # 仅展示第一个 batch，这里 break
#         break


