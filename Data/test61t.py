'''
用于从原始数据集中获得部分预处理后的训练集和内部测试集。
具体来讲，从PASRawData里的center1、center2及其标注里获得数据，筛选配对，划分训练集和测试集，进行部分预处理，最后生成MSD格式的文件。
'''

# 多线程处理，包含预处理步骤
import os
import random
import SimpleITK as sitk
import numpy as np
import json
from skimage import exposure
from tqdm import tqdm
import concurrent.futures

# 设置随机数种子，以确保结果可重复
random.seed(42)

# 定义数据的路径
base_path = "/home/vipuser/Desktop/Data/PASRawData"
centers = ["center1", "center2"]
label_dirs = {"center1": "center1_label", "center2": "center2_label"}

# 定义输出MSD格式的数据集路径
task_name = "Task02_PASp1"
msd_path = os.path.join(base_path, task_name)
imagesTr_path = os.path.join(msd_path, "imagesTr")
labelsTr_path = os.path.join(msd_path, "labelsTr")
imagesTs_path = os.path.join(msd_path, "imagesTs")
labelsTs_path = os.path.join(msd_path, "labelsTs")

# 创建MSD输出目录
for path in [imagesTr_path, labelsTr_path, imagesTs_path, labelsTs_path]:
    os.makedirs(path, exist_ok=True)

# 标准化处理函数 (此函数不再被调用，可以注释掉或删除)
# def preprocess_image(image_data, original_spacing):
#     max_threshold = np.percentile(image_data, 99.5)
#     min_threshold = np.percentile(image_data, 0.5)
#     image_data = np.clip(image_data, min_threshold, max_threshold)
#
#     sitk_image = sitk.GetImageFromArray(image_data)
#     sitk_image.SetSpacing(original_spacing)
#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     image_corrected = corrector.Execute(sitk_image)
#     image_data_corrected = sitk.GetArrayFromImage(image_corrected)
#
#     # image_equalized = exposure.equalize_hist(image_data_corrected)
#
#     return sitk.GetImageFromArray(image_data_corrected)

# 获取影像与标签的匹配文件
def get_matching_files():
    matching_pairs = []
    for center in centers:
        center_path_PAS = os.path.join(base_path, center, 'PAS')
        center_path_NOPAS = os.path.join(base_path, center, 'NoPAS')
        label_path = os.path.join(base_path, label_dirs[center])
        image_files = [f for f in os.listdir(center_path_PAS) if f.endswith(".nii.gz")] + [f for f in os.listdir(center_path_NOPAS) if f.endswith(".nii.gz")]
        label_files = [f for f in os.listdir(label_path) if f.endswith(".nii.gz")]
        image_names = {f.split('_')[0]: f for f in image_files}
        label_names = {f.split('.')[0]: f for f in label_files}
        
        for image_id, image_file in tqdm(image_names.items(), desc=f"Matching files in {center}", leave=False):
            if image_id in label_names:
                image_path = os.path.join(center_path_PAS if os.path.exists(os.path.join(center_path_PAS, image_file)) else center_path_NOPAS, image_file)
                label_path = os.path.join(label_dirs[center], label_names[image_id])
                if os.path.exists(os.path.join(base_path, label_path)):
                    # 读取图像和标签
                    image = sitk.ReadImage(image_path)
                    label = sitk.ReadImage(os.path.join(base_path, label_path))
                    
                    # 检查图像和标签的形状是否一致
                    if image.GetSize() == label.GetSize():
                        matching_pairs.append((image_path, os.path.join(base_path, label_path)))
                    else:
                        print(f"Skipping {image_id} due to shape mismatch.")
    # 仅取前25个和后25个匹配对
    # matching_pairs = matching_pairs[:5] + matching_pairs[-5:]
    print(len(matching_pairs))
    return matching_pairs

# 生成 dataset.json 文件
def generate_dataset_json():
    dataset = {
        "name": task_name,
        "description": "Intelligent Diagnosis of Placenta Accreta in Pregnant Women Based on MRI Image Analysis",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": {"0": "MRI"},
        "labels": {"0": "background", "1": "placenta_accreta"},
        "numTraining": len(os.listdir(labelsTr_path)),
        "numTest": len(os.listdir(imagesTs_path)),
        "training": [{"image": f"./imagesTr/{file}", "label": f"./labelsTr/{file.split('_')[0]}"} for file in os.listdir(imagesTr_path)],
        "test": [{"image": f"./imagesTs/{file}"} for file in os.listdir(imagesTs_path)]
    }
    with open(os.path.join(msd_path, "dataset.json"), "w") as json_file:
        json.dump(dataset, json_file, indent=4)

# 处理图像和标签的函数
def process_image_label_pair(image_path, label_path, output_image_dir, output_label_dir):
    patient_id = os.path.basename(image_path).split('_')[0]
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    original_spacing = image.GetSpacing()
    # 移除预处理步骤:
    # processed_image = preprocess_image(sitk.GetArrayFromImage(image), original_spacing)
    # processed_image.SetSpacing(original_spacing)

    # 确保原始图像和标签具有正确的 Spacing
    image.SetSpacing(original_spacing)
    label.SetSpacing(original_spacing) # 确保标签也有正确的 Spacing

    # 保存原始图像和标签
    sitk.WriteImage(image, os.path.join(output_image_dir, f"{patient_id}.nii.gz")) # 保存原始 image
    sitk.WriteImage(label, os.path.join(output_label_dir, f"{patient_id}.nii.gz"))

# 使用并行化加速文件处理并显示进度
def copy_files_to_msd(matching_pairs):
    random.shuffle(matching_pairs)
    split_index = int(len(matching_pairs) * 0.8)
    training_pairs = matching_pairs[:split_index]
    test_pairs = matching_pairs[split_index:]

    # 设置进度条
    total_tasks = len(training_pairs) + len(test_pairs)
    with tqdm(total=total_tasks, desc="Processing pairs", unit="task") as pbar:
        # 并行化训练和测试数据处理
        def update_progress(future):
            pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            # 提交训练数据任务
            for image_path, label_path in training_pairs:
                future = executor.submit(process_image_label_pair, image_path, label_path, imagesTr_path, labelsTr_path)
                future.add_done_callback(update_progress)
                futures.append(future)
            
            # 提交测试数据任务
            for image_path, label_path in test_pairs:
                future = executor.submit(process_image_label_pair, image_path, label_path, imagesTs_path, labelsTs_path)
                future.add_done_callback(update_progress)
                futures.append(future)

            # 等待所有任务完成
            concurrent.futures.wait(futures)

# 主函数
def main():
    matching_pairs = get_matching_files()
    copy_files_to_msd(matching_pairs)
    generate_dataset_json()

if __name__ == "__main__":
    main()
