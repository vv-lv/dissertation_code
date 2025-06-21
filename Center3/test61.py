'''
用于将center3的MRI图像文件预处理后复制到Task02_PASp61/imagesTs_3目录下。
包含与原始处理程序相同的预处理步骤。
'''

import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import concurrent.futures

# 定义数据的路径
base_path = "/home/vipuser/Desktop/Data/PASRawData"
center3_path = os.path.join(base_path, "center3")

# 定义输出目录
msd_path = '/home/vipuser/Desktop/Data/Task02_PASp61'
imagesTs_3_path = os.path.join(msd_path, "imagesTs_3")

# 创建输出目录
os.makedirs(imagesTs_3_path, exist_ok=True)

# 标准化处理函数 - 与原始程序相同
def preprocess_image(image_data, original_spacing):
    # 裁剪极值
    max_threshold = np.percentile(image_data, 99.5)
    min_threshold = np.percentile(image_data, 0.5)
    image_data = np.clip(image_data, min_threshold, max_threshold)

    # 转换为SimpleITK图像并设置原始间距
    sitk_image = sitk.GetImageFromArray(image_data)
    sitk_image.SetSpacing(original_spacing)
    
    # 应用N4偏置场校正
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    image_corrected = corrector.Execute(sitk_image)
    image_data_corrected = sitk.GetArrayFromImage(image_corrected)

    # 返回处理后的图像
    return sitk.GetImageFromArray(image_data_corrected)

# 处理单个图像文件
def process_image(image_path):
    try:
        patient_id = os.path.basename(image_path).split('_')[0]
        
        # 读取原始图像
        image = sitk.ReadImage(image_path)
        original_spacing = image.GetSpacing()
        
        # 应用预处理
        image_array = sitk.GetArrayFromImage(image)
        processed_image = preprocess_image(image_array, original_spacing)
        processed_image.SetSpacing(original_spacing)
        
        # 保存处理后的图像
        output_path = os.path.join(imagesTs_3_path, f"{patient_id}.nii.gz")
        sitk.WriteImage(processed_image, output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_center3_images():
    # 获取center3中PAS和NoPAS文件夹中的所有nii.gz文件
    center3_pas_path = os.path.join(center3_path, 'PAS')
    center3_nopas_path = os.path.join(center3_path, 'NoPAS')
    
    pas_files = []
    if os.path.exists(center3_pas_path):
        pas_files = [os.path.join(center3_pas_path, f) for f in os.listdir(center3_pas_path) 
                    if f.endswith(".nii.gz")]
    
    nopas_files = []
    if os.path.exists(center3_nopas_path):
        nopas_files = [os.path.join(center3_nopas_path, f) for f in os.listdir(center3_nopas_path) 
                      if f.endswith(".nii.gz")]
    
    # 合并所有文件
    all_files = pas_files + nopas_files
    print(f"Found {len(all_files)} image files in center3")
    
    # 使用并行处理来加速预处理过程
    successful = 0
    with tqdm(total=len(all_files), desc="Processing images") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file_path in all_files:
                future = executor.submit(process_image, file_path)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful += 1
                pbar.update(1)
    
    print(f"Successfully processed {successful} out of {len(all_files)} files to {imagesTs_3_path}")

def main():
    process_center3_images()
    print("Task completed!")

if __name__ == "__main__":
    main()

# import os
# import SimpleITK as sitk
# import numpy as np
# from tqdm import tqdm

# # 定义数据的路径
# base_path = "/home/vipuser/Desktop/Data/PASRawData"
# center3_path = os.path.join(base_path, "center3")

# # 定义输出目录
# msd_path = '/home/vipuser/Desktop/Data/Task02_PASp61'
# imagesTs_3_path = os.path.join(msd_path, "imagesTs_3")

# # 创建输出目录
# os.makedirs(imagesTs_3_path, exist_ok=True)

# def preprocess_image(image_data, original_spacing):
#     max_threshold = np.percentile(image_data, 99.5)
#     min_threshold = np.percentile(image_data, 0.5)
#     image_data = np.clip(image_data, min_threshold, max_threshold)

#     sitk_image = sitk.GetImageFromArray(image_data)
#     sitk_image.SetSpacing(original_spacing)

#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     image_corrected = corrector.Execute(sitk_image)
#     image_data_corrected = sitk.GetArrayFromImage(image_corrected)

#     return sitk.GetImageFromArray(image_data_corrected)

# def process_image(image_path):
#     try:
#         patient_id = os.path.basename(image_path).split('_')[0]
        
#         # 读取原始图像
#         image = sitk.ReadImage(image_path)
#         original_spacing = image.GetSpacing()
        
#         # 应用预处理
#         image_array = sitk.GetArrayFromImage(image)
#         processed_image = preprocess_image(image_array, original_spacing)
#         processed_image.SetSpacing(original_spacing)
        
#         # 保存处理后的图像
#         output_path = os.path.join(imagesTs_3_path, f"{patient_id}.nii.gz")
#         sitk.WriteImage(processed_image, output_path)
#         return True
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return False

# def process_center3_images():
#     center3_pas_path = os.path.join(center3_path, 'PAS')
#     center3_nopas_path = os.path.join(center3_path, 'NoPAS')

#     pas_files = []
#     if os.path.exists(center3_pas_path):
#         pas_files = [os.path.join(center3_pas_path, f) for f in os.listdir(center3_pas_path) 
#                      if f.endswith(".nii.gz")]

#     nopas_files = []
#     if os.path.exists(center3_nopas_path):
#         nopas_files = [os.path.join(center3_nopas_path, f) for f in os.listdir(center3_nopas_path) 
#                        if f.endswith(".nii.gz")]

#     all_files = pas_files + nopas_files
#     print(f"Found {len(all_files)} image files in center3")

#     successful = 0
#     with tqdm(total=len(all_files), desc="Processing images") as pbar:
#         # 串行逐个文件处理，不再并行
#         for file_path in all_files:
#             if process_image(file_path):
#                 successful += 1
#             pbar.update(1)

#     print(f"Successfully processed {successful} out of {len(all_files)} files to {imagesTs_3_path}")

# def main():
#     process_center3_images()
#     print("Task completed!")

# if __name__ == "__main__":
#     main()




