import os
import glob
import pandas as pd

def create_labels_file(features_csv_path, output_label_path, base_data_path):
    """
    根据特征文件中的ID创建对应的标签文件
    
    参数:
    features_csv_path - 特征CSV文件路径
    output_label_path - 输出标签CSV文件路径
    base_data_path - 原始数据根目录路径
    """
    # 读取特征文件，获取ID列，并确保ID为字符串类型
    features_df = pd.read_csv(features_csv_path)
    features_df['ID'] = features_df['ID'].astype(str)  # 将ID转换为字符串类型
    ids = features_df['ID'].tolist()
    
    labels = []
    skipped_ids = []
    
    for file_identifier in ids:
        # 有些ID可能有.nii后缀，确保移除
        if file_identifier.endswith('.nii'):
            file_identifier = file_identifier[:-4]
            
        # 搜索NoPAS和PAS目录
        no_pas_match = (
            glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file_identifier + '*.nii.gz')) or
            glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file_identifier + '*.nii.gz'))
        )
        
        pas_match = (
            glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file_identifier + '*.nii.gz')) or
            glob.glob(os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file_identifier + '*.nii.gz'))
        )
        
        if no_pas_match:
            labels.append(0)
        elif pas_match:
            labels.append(1)
        else:
            print(f"警告：无法在PAS/NoPAS目录中找到文件 {file_identifier}，将跳过")
            skipped_ids.append(file_identifier)
            labels.append(None)  # 或者您可以选择其他默认值
    
    # 创建标签DataFrame
    label_df = pd.DataFrame({
        'ID': ids,
        'label': labels
    })
    
    # 删除找不到标签的行
    if skipped_ids:
        print(f"找不到标签的ID数量: {len(skipped_ids)}")
        label_df = label_df.dropna(subset=['label'])
    
    # 保存标签文件
    label_df.to_csv(output_label_path, index=False)
    print(f"标签文件已保存到 {output_label_path}")
    print(f"总记录数: {len(label_df)}")
    print(f"标签分布: {label_df['label'].value_counts().to_dict()}")
    
    return label_df

# 使用示例
if __name__ == "__main__":
    # 特征文件路径
    train_features_path = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/train_features.csv"
    test_features_path = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_features.csv"
    
    # 输出标签文件路径
    train_labels_path = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/train_labels.csv"
    test_labels_path = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_labels.csv"
    
    # 原始数据根目录
    base_path = "/home/vipuser/Desktop/Data/Task02_PASp62"
    
    # 创建训练集标签
    print("创建训练集标签...")
    train_labels = create_labels_file(train_features_path, train_labels_path, base_path)
    
    # 创建测试集标签
    print("\n创建测试集标签...")
    test_labels = create_labels_file(test_features_path, test_labels_path, base_path)