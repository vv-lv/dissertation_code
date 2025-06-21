import os
import glob
import numpy as np
import torch
import SimpleITK as sitk

# ---------------------------
# 1. 定义 Dataset 类（用于 3D 医学图像分类任务）
# ---------------------------
class NiiDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        image_dir: 存放 NIfTI 图像 (.nii.gz) 的文件夹路径
        transform: 使用 Batchgeneratorsv2 构建的增强流水线，输入输出均为关键字参数形式
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        # 分类任务：示例中随机生成标签；实际请根据数据进行调整
        self.labels = [np.random.randint(0, 1) for _ in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[idx])).astype(np.float32)
        label = self.labels[idx]
        # image = image[None, ...]

        data_dict = {'image': image, 'label': label}
        
        if self.transform:
            data_dict = self.transform(**data_dict)
        
        # # 可选：确保变换后仍然有通道维度
        # if data_dict['image'].ndim == 3:
        #     data_dict['image'] = data_dict['image'][None, ...]
        
        return data_dict



# ---------------------------
# 2. 定义 GammaTransformWrapper
# ---------------------------
class TensorWrapper:
    """
    包装 GammaTransform，使其能接收 numpy 数组：
      - 将输入 image 转换为 torch.Tensor，
      - 调用 GammaTransform，
      - 将结果转换回 numpy 数组
    """
    def __init__(self, gamma_transform):
        self.gamma_transform = gamma_transform

    def __call__(self, **kwargs):
        image = kwargs.get('image')
        # 若输入不是 tensor，则转换
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        kwargs['image'] = image
        # 调用 GammaTransform
        result = self.gamma_transform(**kwargs)
        # 转换回 numpy 数组
        if isinstance(result['image'], torch.Tensor):
            result['image'] = result['image'].cpu().numpy()
        return result

# ---------------------------
# 3. 构建数据增强流水线（使用 Batchgeneratorsv2）
# ---------------------------
# 正确导入 Compose（v0.2.1 版本中 Compose 位于 batchgeneratorsv2.transforms.compose 模块）
from batchgenerators.transforms.abstract_transforms import Compose

from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

# from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform

# 定义输出 patch 大小（此处设为固定尺寸，可根据需要修改）
patch_size = (35, 255, 255)
rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)

transforms_list = []


ignore_axes = (0,)
transforms_list.append(Convert3DTo2DTransform())
patch_size_spatial = patch_size[1:]

# 若需要空间变换，可恢复此块（同步变换 label 时可设置 label_key='label'）
# spatial_transform = SpatialTransform(
#     patch_size=patch_size,  # 例如 (80, 160, 160)
#     patch_center_dist_from_border=0,
#     do_elastic_deform=True,
#     alpha=(0., 900.),
#     sigma=(9., 13.),
#     do_rotation=True,
#     angle_x=(-15/360*2*np.pi, 15/360*2*np.pi),
#     angle_y=(-15/360*2*np.pi, 15/360*2*np.pi),
#     angle_z=(-15/360*2*np.pi, 15/360*2*np.pi),
#     do_scale=True,
#     scale=(0.85, 1.25),
#     border_mode_data='nearest',
#     border_cval_data=0,
#     order_data=3,
#     border_mode_seg='constant',
#     border_cval_seg=0,
#     order_seg=0,
#     random_crop=False,
#     data_key='image',
#     label_key=None,
#     p_el_per_sample=0.2,
#     p_rot_per_sample=0.2,
#     p_scale_per_sample=0.2,
#     independent_scale_for_each_axis=False,  # 同步缩放
#     p_rot_per_axis=1, 
#     p_independent_scale_per_axis=1
# )
spatial_transform = SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
transforms_list.append(spatial_transform)
    # angle_x=(-15/360*2*np.pi, 15/360*2*np.pi), angle_y=(-15/360*2*np.pi, 15/360*2*np.pi), angle_z=(-15/360*2*np.pi, 15/360*2*np.pi),
transforms_list.append(Convert2DTo3DTransform())

# 2.2 高斯噪声：添加随机噪声
noise_transform = RandomTransform(
    GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
    apply_probability=0.1
)
transforms_list.append(noise_transform)

# 2.3 高斯模糊：模拟成像模糊
blur_transform = RandomTransform(
    GaussianBlurTransform(blur_sigma=(0.5, 1.0), synchronize_channels=False, synchronize_axes=False, p_per_channel=0.5, benchmark=True),
    apply_probability=0.2
)
transforms_list.append(blur_transform)

# 2.4 亮度调整：乘法因子调整亮度
brightness_transform = RandomTransform(
    MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)), synchronize_channels=False, p_per_channel=1),
    apply_probability=0.15
)
transforms_list.append(brightness_transform)

# 2.5 对比度调整
contrast_transform = RandomTransform(
    ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True, synchronize_channels=False, p_per_channel=1),
    apply_probability=0.15
)
transforms_list.append(contrast_transform)

# 2.6 伽马校正：分 invert 与 non-invert 两种版本
gamma_transform_invert = RandomTransform(
                TensorWrapper(GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            )),
    apply_probability=0.1
)
transforms_list.append(gamma_transform_invert)

gamma_transform_normal = RandomTransform(
                TensorWrapper(GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            )),
    apply_probability=0.3
)
transforms_list.append(gamma_transform_normal)

# 2.7 镜像翻转：随机沿各轴翻转
mirror_transform = TensorWrapper(MirrorTransform(allowed_axes=(0, 1, 2)))
transforms_list.append(mirror_transform)

# 2.8 转换为 Tensor（确保数据格式适用于 PyTorch），键设置为 ["image", "label"]
numpy_to_tensor = NumpyToTensor(["image", "label"], "float")
transforms_list.append(numpy_to_tensor)

# 组合所有 transform
aug_pipeline = Compose(transforms_list)

# ---------------------------
# 3. 定义自定义数据加载器，包装 PyTorch DataLoader
# ---------------------------
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

class MyDataLoader(SlimDataLoaderBase):
    def __init__(self, loader):
        self.loader = loader
        self._iterator = iter(self.loader)
        self.batch_size = loader.batch_size
        self.num_samples = len(loader.dataset)

    def generate_train_batch(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        return batch

    def set_thread_id(self, thread_id):
        pass

# ---------------------------
# 4. 数据加载与多线程增强
# ---------------------------
train_image_dir = '/home/vipuser/Desktop/Data/Task02_PASp61/imagesTr'
train_dataset = NiiDataset(train_image_dir, transform=aug_pipeline)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=0
)

my_data_loader = MyDataLoader(train_loader)

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

mt_augmenter = MultiThreadedAugmenter(
    data_loader=my_data_loader,
    transform=aug_pipeline,
    num_processes=1,  # 测试时先用单进程，确认无误后可调整为多进程
    num_cached_per_queue=1,
)

# ---------------------------
# 5. 示例训练循环（获取几个批次数据）
# ---------------------------
for i, batch in enumerate(mt_augmenter):
    # batch 为字典，包含 "image" 和 "label"
    images = batch["image"]   # 形状应为 (B, C, D, H, W)
    labels = batch["label"]
    
    print(f"Batch {i}: images shape {images.shape}, labels shape {labels.shape}")
    
    # NumpyToTensor 已转换为 PyTorch tensor，若需要可再次封装
    images_torch = torch.tensor(images, dtype=torch.float32)
    labels_torch = torch.tensor(labels, dtype=torch.long)
    
    # 此处可进行前向传播、损失计算、反向传播等训练步骤
    # ...
    
    if i >= 2:
        break

# 结束训练后关闭增强器，释放子进程
mt_augmenter.shutdown()
