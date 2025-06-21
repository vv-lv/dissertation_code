# CLAHE相关算法测试
import numpy as np
import SimpleITK as sitk
import cv2

# 标准化处理函数
def preprocess_image(image_data, original_spacing, gamma_values=[0.7, 0.85, 1.25, 1.5], clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对输入图像数据进行预处理，包括：
    - N4偏置场校正
    - Gamma 和 CLAHE 混合增强替代直方图均衡化

    参数：
        - image_data: 输入的图像数据（3D 数组）。
        - original_spacing: 图像的原始像素间距。
        - gamma_values: Gamma 增强的值列表。
        - clip_limit: CLAHE 算法的剪切限制。
        - tile_grid_size: CLAHE 算法的网格大小。
    返回：
        - 经过预处理的 SimpleITK 图像。
    """
    # 第一步：对强度进行百分位裁剪
    max_threshold = np.percentile(image_data, 99.7)
    image_data = np.clip(image_data, 0, max_threshold)

    # 第二步：N4偏置场校正
    sitk_image = sitk.GetImageFromArray(image_data)
    sitk_image.SetSpacing(original_spacing)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    image_corrected = corrector.Execute(sitk_image)
    image_data_corrected = sitk.GetArrayFromImage(image_corrected)

    # 第三步：使用混合 Gamma 和 CLAHE 增强亮度分量
    image_data_enhanced = apply_gamma_clahe(image_data_corrected, gamma_values, clip_limit, tile_grid_size)

    # 返回增强后的图像
    return sitk.GetImageFromArray(image_data_enhanced)


def apply_gamma_clahe(image_data, gamma_values, clip_limit, tile_grid_size):
    """
    对 3D 医学图像数据应用混合 Gamma 和 CLAHE 算法进行增强。
    
    参数：
        - image_data: 输入的 3D 图像数据。
        - gamma_values: 一组 Gamma 值。
        - clip_limit: CLAHE 剪切限制。
        - tile_grid_size: CLAHE 网格大小。
    
    返回：
        - 增强后的 3D 图像数据。
    """
    enhanced_volume = np.zeros_like(image_data, dtype=np.float32)

    for i in range(image_data.shape[0]):  # 遍历每个切片
        slice_ = image_data[i, :, :]
        
        # 归一化到 [0, 1]
        slice_normalized = slice_ / np.max(slice_)
        
        # 应用 Gamma 增强
        gamma_results = []
        for gamma in gamma_values:
            gamma_corrected = np.power(slice_normalized, gamma)
            gamma_results.append(gamma_corrected)
        
        # 计算对比度，选择最佳 Gamma 增强结果
        contrast_values = [np.std(g) for g in gamma_results]
        best_gamma_index = np.argmax(contrast_values)
        best_gamma_image = gamma_results[best_gamma_index]
        
        # 应用 CLAHE 增强
        slice_uint8 = np.uint8(best_gamma_image * 255)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_slice = clahe.apply(slice_uint8)
        
        # 归一化回 [0, 1]
        enhanced_slice = enhanced_slice / 255.0
        enhanced_volume[i, :, :] = enhanced_slice

    return enhanced_volume



# 示例图像数据（3D 数组）
image_data = np.random.rand(50, 256, 256)  # 示例大小为 (50, 256, 256)
original_spacing = [1.0, 1.0, 1.0]

# 调用预处理函数
preprocessed_image = preprocess_image(image_data, original_spacing)

# # 保存预处理后的图像
# sitk.WriteImage(preprocessed_image, "preprocessed_image.nii")
