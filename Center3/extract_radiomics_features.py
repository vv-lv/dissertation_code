import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor

def preprocess_images(image_dir, mask_dir, output_image_dir, output_mask_dir, margin=(2, 20, 20)):
    """
    使用SimpleITK对图像和掩膜进行预处理，裁剪前景区域
    """
    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # 获取文件列表
    file_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    total_files = len(file_list)
    
    print(f"找到 {total_files} 个.nii.gz文件")
    
    processed_count = 0
    for idx, fname in enumerate(file_list):
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        if not os.path.exists(mask_path):
            print(f"警告：找不到 {fname} 对应的掩膜，跳过！")
            continue
        
        # 明确指定输出文件路径
        output_image_path = os.path.join(output_image_dir, fname)
        output_mask_path = os.path.join(output_mask_dir, fname)
        
        try:
            # 读取图像和掩膜
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            
            # 将掩膜转换为二值图像
            mask_binary = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255)
            
            # 计算掩膜的边界框
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(mask_binary)
            
            if label_shape_filter.GetNumberOfLabels() == 0:
                print(f"警告：{fname} 的掩膜中没有前景，保存原始图像")
                sitk.WriteImage(image, output_image_path)
                sitk.WriteImage(mask, output_mask_path)
                processed_count += 1
                continue
            
            # 获取边界框（针对3D图像）
            bounding_box = label_shape_filter.GetBoundingBox(1)  # 1是前景标签
            x_start, y_start, z_start = bounding_box[0], bounding_box[1], bounding_box[2]
            x_size, y_size, z_size = bounding_box[3], bounding_box[4], bounding_box[5]
            
            # 添加边距
            x_start = max(0, x_start - margin[2])
            y_start = max(0, y_start - margin[1])
            z_start = max(0, z_start - margin[0])
            
            x_end = min(image.GetSize()[0], x_start + x_size + 2 * margin[2])
            y_end = min(image.GetSize()[1], y_start + y_size + 2 * margin[1])
            z_end = min(image.GetSize()[2], z_start + z_size + 2 * margin[0])
            
            # 重新计算大小
            size = [x_end - x_start, y_end - y_start, z_end - z_start]
            
            # 裁剪图像和掩膜
            extract_filter = sitk.ExtractImageFilter()
            extract_filter.SetSize(size)
            extract_filter.SetIndex([x_start, y_start, z_start])
            
            cropped_image = extract_filter.Execute(image)
            cropped_mask = extract_filter.Execute(mask)
            
            # 保存裁剪后的图像和掩膜
            sitk.WriteImage(cropped_image, output_image_path)
            sitk.WriteImage(cropped_mask, output_mask_path)
            processed_count += 1
            
        except Exception as e:
            print(f"处理 {fname} 时出错: {str(e)}")
            # 尝试保存原始图像
            try:
                sitk.WriteImage(image, output_image_path)
                sitk.WriteImage(mask, output_mask_path)
                processed_count += 1
                print(f"已保存原始未裁剪图像: {output_image_path}")
            except Exception as save_error:
                print(f"无法保存 {fname}: {str(save_error)}")
        
        if (idx + 1) % 10 == 0 or idx == total_files - 1:
            print(f"处理进度: {idx+1}/{total_files} ({processed_count} 个成功)")
    
    print(f"预处理完成，共处理 {processed_count}/{total_files} 个文件")
    
    # 验证输出目录中的文件
    output_files = [f for f in os.listdir(output_image_dir) if f.endswith('.nii.gz')]
    print(f"输出目录包含 {len(output_files)} 个处理后的图像文件")
    
    return output_image_dir, output_mask_dir

def extract_features_from_folder(image_dir, mask_dir):
    """
    遍历指定文件夹下所有图像，并与对应的掩膜配对提取影像组学特征
    """
    # 创建特征提取器
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # 设置灰度分箱宽度，不做重采样和归一化
    extractor.settings['binWidth'] = 25
    extractor.settings['resampledPixelSpacing'] = None  # 使用原始体素尺寸
    
    # 删除插值设置（不需要重采样时可以删除该参数）
    if 'interpolator' in extractor.settings:
        del extractor.settings['interpolator']
    
    # 启用所有滤波类型
    extractor.enableImageTypes(
        Original={},
        LoG={'sigma': [1.0, 2.0, 3.0]},    # Laplacian of Gaussian滤波
        Wavelet={},                       # 小波滤波
        # Square={}, SquareRoot={},         # 平方、平方根滤波
        # Logarithm={}, Exponential={},     # 对数、指数滤波
        # Gradient={},                      # 梯度滤波
        # LBP3D={'binWidth': 1.0}           # 三维LBP滤波
    )
    extractor.enableAllFeatures()
    
    features_list = []
    file_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    total_files = len(file_list)
    
    print(f"开始提取 {total_files} 个文件的特征...")
    
    for idx, fname in enumerate(file_list):
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        if not os.path.exists(mask_path):
            print(f"警告：找不到 {fname} 对应的掩膜，跳过！")
            continue
        
        try:
            # 读取影像和掩膜
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            
            # 验证图像与掩膜
            validate_image_mask(image, mask)
            
            # 执行特征提取
            result = extractor.execute(image, mask)
            
            # 去除诊断信息，只保留实际特征
            result = {k: v for k, v in result.items() if not k.startswith("diagnostics")}
            
            # 添加文件标识（去除扩展名）
            base_name = os.path.basename(fname)
            if base_name.endswith('.nii.gz'):
                result['ID'] = base_name[:-7]  # 去除.nii.gz
            else:
                result['ID'] = os.path.splitext(base_name)[0]
                
            features_list.append(result)
            
            if (idx + 1) % 10 == 0 or idx == total_files - 1:
                print(f"特征提取进度: {idx+1}/{total_files}")
                
        except Exception as e:
            print(f"处理 {fname} 时出错: {str(e)}")
    
    if not features_list:
        print("警告：没有成功提取任何特征！")
        return pd.DataFrame()
    
    # 创建DataFrame
    df = pd.DataFrame(features_list)
    
    # 确保ID列在最前面
    if 'ID' in df.columns:
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]
    
    # 计算特征数量
    feature_count = len(df.columns) - 1  # 减去ID列
    
    print(f"特征提取完成，共提取 {len(df)} 个样本的 {feature_count} 个特征")
    
    return df

def validate_image_mask(image, mask, min_roi_size=100):
    """验证图像和掩膜的兼容性以及ROI大小"""
    # 检查尺寸是否匹配
    if image.GetSize() != mask.GetSize():
        raise ValueError(f"尺寸不匹配：图像 {image.GetSize()}，掩膜 {mask.GetSize()}")
    
    # 检查掩膜是否包含足够的体素
    mask_array = sitk.GetArrayFromImage(mask)
    roi_voxels = np.sum(mask_array > 0)
    if roi_voxels < min_roi_size:
        print(f"警告：ROI较小，仅有 {roi_voxels} 个体素")
    
    # 检查像素类型，确保适合特征提取
    if image.GetPixelID() not in [sitk.sitkFloat32, sitk.sitkFloat64]:
        print(f"注意：图像类型为 {image.GetPixelIDTypeAsString()}，非浮点型可能影响特征提取准确性")
    
    return True

def main():
    # 设置数据集根目录，请修改为实际路径
    base_dir = "/home/vipuser/Desktop/Data/Task02_PASp62"
    output_dir = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # # 训练集和测试集对应的图像和掩膜文件夹
    # imagesTr_dir = os.path.join(base_dir, "imagesTr")
    # labelsTr_dir = os.path.join(base_dir, "labelsTr")
    imagesTs_dir = os.path.join(base_dir, "imagesTs_3")
    labelsTs_dir = os.path.join(base_dir, "labelsTs_3")
    
    # 预处理后的输出目录
    # processed_imagesTr_dir = os.path.join(output_dir, "imagesTr")
    # processed_labelsTr_dir = os.path.join(output_dir, "labelsTr")
    processed_imagesTs_dir = os.path.join(output_dir, "imagesTs_3")
    processed_labelsTs_dir = os.path.join(output_dir, "labelsTs_3")
    
    # 预处理训练集
    # print("="*50)
    # print("开始预处理训练集...")
    # preprocess_images(imagesTr_dir, labelsTr_dir, processed_imagesTr_dir, processed_labelsTr_dir, margin=(2, 20, 20))
    
    # 预处理测试集
    print("="*50)
    print("开始预处理测试集...")
    preprocess_images(imagesTs_dir, labelsTs_dir, processed_imagesTs_dir, processed_labelsTs_dir, margin=(2, 20, 20))
    
    
    # # 提取训练集特征
    # print("="*50)
    # print("开始提取训练集特征...")
    # train_df = extract_features_from_folder(processed_imagesTr_dir, processed_labelsTr_dir)
    # train_csv = os.path.join(output_dir, "train_features.csv")
    # train_df.to_csv(train_csv, index=False)
    # print(f"训练集特征已保存到 {train_csv}")
    
    # 提取测试集特征
    print("="*50)
    print("开始提取测试集特征...")
    test_df = extract_features_from_folder(processed_imagesTs_dir, processed_labelsTs_dir)
    test_csv = os.path.join(output_dir, "test_features_3.csv")
    test_df.to_csv(test_csv, index=False)
    print(f"测试集特征已保存到 {test_csv}")
    
    print("="*50)
    print("所有处理完成！")

if __name__ == "__main__":
    main()