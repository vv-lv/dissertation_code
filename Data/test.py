# import os
# import glob
# import nibabel as nib
# import numpy as np

# # 设置包含 .nii.gz 文件的文件夹路径
# folder_path = "/home/vipuser/Desktop/Data/Task02_PASp62_local/imagesTr"  # 修改为你的文件夹路径
# file_paths = glob.glob(os.path.join(folder_path, "*.nii.gz"))

# shapes = []   # 用于存储每个文件的 shape
# file_list = []  # 存储文件名（可选）

# for fp in file_paths:
#     try:
#         img = nib.load(fp)
#         data = img.get_fdata()
#         data = np.transpose(data, (2, 0, 1))
#         # 如果数据是4D（例如多通道或时间序列），这里只取前三个维度
#         if data.ndim >= 3:
#             shape = data.shape[:3]
#         else:
#             shape = data.shape
#         shapes.append(shape)
#         file_list.append(os.path.basename(fp))
#     except Exception as e:
#         print(f"加载 {fp} 时出错：{e}")

# shapes = np.array(shapes)
# num_files = shapes.shape[0]
# print(f"共找到 {num_files} 个文件。")

# if num_files == 0:
#     print("没有找到任何文件。")
# else:
#     # 假设所有图像都是3D的，统计每个维度的信息
#     dims = shapes.shape[1]  # 一般为3（X, Y, Z）
#     for i in range(dims):
#         dim_values = shapes[:, i]
#         print(f"\n维度 {i+1}（第 {i} 个索引）的统计信息：")
#         print("  最小值: ", np.min(dim_values))
#         print("  最大值: ", np.max(dim_values))
#         print("  平均值: ", np.mean(dim_values))
#         print("  中值:   ", np.median(dim_values))
#         print("  标准差: ", np.std(dim_values))
    
#     # # 如果需要，也可以输出每个文件的 shape 详情
#     # print("\n各文件的 shape：")
#     # for fn, s in zip(file_list, shapes):
#     #     print(f"  {fn}: {s}")




# import os
# import glob
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl


# def get_bounding_box(label_path):
#     """
#     加载 .nii.gz 标签文件，并返回其非零区域的 3D bounding box，
#     格式为 (z_min, y_min, x_min, z_max, y_max, x_max)。
#     若文件中无前景，则返回整个图像的边界。
#     """
#     # 加载 nii.gz 文件
#     label_img = nib.load(label_path)
#     label_data = label_img.get_fdata().astype(np.uint8)
#     # 转置为 (D, H, W) 格式，其中 D 为深度，H 为高度，W 为宽度
#     label_data = np.transpose(label_data, (2, 0, 1))

#     D, H, W = label_data.shape
    
#     # 查找所有非零体素的坐标
#     coords = np.argwhere(label_data != 0)
#     if coords.size == 0:
#         return (0, 0, 0, D - 1, H - 1, W - 1)
    
#     z_min, y_min, x_min = coords[:, 0].min(), coords[:, 1].min(), coords[:, 2].min()
#     z_max, y_max, x_max = coords[:, 0].max(), coords[:, 1].max(), coords[:, 2].max()

#     # 双向扩展：在每个维度上分别将最小值减 margin，最大值加 margin，并保证不超出图像边界
#     margin = (2, 20, 20)
#     z_min = max(z_min - margin[0], 0)
#     y_min = max(y_min - margin[1], 0)
#     x_min = max(x_min - margin[2], 0)
    
#     z_max = min(z_max + margin[0], D - 1)
#     y_max = min(y_max + margin[1], H - 1)
#     x_max = min(x_max + margin[2], W - 1)
#     return (int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max))

# def main(folder_path):
#     # 获取指定文件夹下所有 .nii.gz 文件
#     file_list = glob.glob(os.path.join(folder_path, '*.nii.gz'))
    
#     # 存储每个维度的bounding box长度（定义为：max - min + 1）
#     z_lengths = []
#     y_lengths = []
#     x_lengths = []
    
#     for file in file_list:
#         bbox = get_bounding_box(file)
#         z_min, y_min, x_min, z_max, y_max, x_max = bbox
#         # 计算每个维度的长度
#         z_length = z_max - z_min + 1
#         y_length = y_max - y_min + 1
#         x_length = x_max - x_min + 1
        
#         z_lengths.append(z_length)
#         y_lengths.append(y_length)
#         x_lengths.append(x_length)
    
#     # 计算每个维度的90%分位点
#     z_90 = np.percentile(z_lengths, 90)
#     y_90 = np.percentile(y_lengths, 90)
#     x_90 = np.percentile(x_lengths, 90)
    
#     print("Z轴(深度) 90%分位点:", z_90)
#     print("Y轴(高度) 90%分位点:", y_90)
#     print("X轴(宽度) 90%分位点:", x_90)
    
#     # 绘制每个维度长度的分布直方图
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
#     # Z轴分布
#     axs[0].hist(z_lengths, bins=20, color='skyblue', edgecolor='black')
#     axs[0].axvline(z_90, color='red', linestyle='dashed', linewidth=2, label=f'90%: {z_90}')
#     axs[0].set_title("Z轴长度分布")
#     axs[0].set_xlabel("长度")
#     axs[0].set_ylabel("文件数量")
#     axs[0].legend()
    
#     # Y轴分布
#     axs[1].hist(y_lengths, bins=20, color='lightgreen', edgecolor='black')
#     axs[1].axvline(y_90, color='red', linestyle='dashed', linewidth=2, label=f'90%: {y_90}')
#     axs[1].set_title("Y轴长度分布")
#     axs[1].set_xlabel("长度")
#     axs[1].set_ylabel("文件数量")
#     axs[1].legend()
    
#     # X轴分布
#     axs[2].hist(x_lengths, bins=20, color='salmon', edgecolor='black')
#     axs[2].axvline(x_90, color='red', linestyle='dashed', linewidth=2, label=f'90%: {x_90}')
#     axs[2].set_title("X轴长度分布")
#     axs[2].set_xlabel("长度")
#     axs[2].set_ylabel("文件数量")
#     axs[2].legend()
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('output.png')  # 保存到当前工作目录
#     print("图像已保存为 output.png")

# if __name__ == "__main__":
#     # 修改为你的 nii.gz 文件所在的文件夹路径
    
#     folder_path = "/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTs_3"
#     main(folder_path)




import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(folder_path):
    # 1) 获取指定文件夹下所有 .nii.gz 文件
    file_list = glob.glob(os.path.join(folder_path, '*.nii.gz'))
    
    # 2) 分别存储每个维度的数据
    z_dims = []
    y_dims = []
    x_dims = []
    
    for file in file_list:
        # 2.1) 加载 .nii.gz 文件
        img = nib.load(file)
        data = img.get_fdata()
        
        # 2.2) 转为 (Z, Y, X) 形状（可视情况省略或调整）
        # 一般 nibabel 读出的顺序可能是 (X, Y, Z)
        data = np.transpose(data, (2, 0, 1))  
        
        # 2.3) 获取当前图像尺寸
        D, H, W = data.shape
        
        # 2.4) 分别记录下来
        z_dims.append(D)
        y_dims.append(H)
        x_dims.append(W)
    
    # 3) 分别计算每个维度的 90% 分位点
    z_90 = np.percentile(z_dims, 90)
    y_90 = np.percentile(y_dims, 90)
    x_90 = np.percentile(x_dims, 90)
    
    print("Z轴(深度) 90%分位点:", z_90)
    print("Y轴(高度) 90%分位点:", y_90)
    print("X轴(宽度) 90%分位点:", x_90)
    
    # 4) 绘制每个维度长度的分布直方图
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 4.1) Z轴分布
    axs[0].hist(z_dims, bins=20, color='skyblue', edgecolor='black')
    axs[0].axvline(z_90, color='red', linestyle='dashed', linewidth=2,
                   label=f'90%: {z_90:.2f}')
    axs[0].set_title("Z轴维度分布")
    axs[0].set_xlabel("长度")
    axs[0].set_ylabel("文件数量")
    axs[0].legend()
    
    # 4.2) Y轴分布
    axs[1].hist(y_dims, bins=20, color='lightgreen', edgecolor='black')
    axs[1].axvline(y_90, color='red', linestyle='dashed', linewidth=2,
                   label=f'90%: {y_90:.2f}')
    axs[1].set_title("Y轴维度分布")
    axs[1].set_xlabel("长度")
    axs[1].set_ylabel("文件数量")
    axs[1].legend()
    
    # 4.3) X轴分布
    axs[2].hist(x_dims, bins=20, color='salmon', edgecolor='black')
    axs[2].axvline(x_90, color='red', linestyle='dashed', linewidth=2,
                   label=f'90%: {x_90:.2f}')
    axs[2].set_title("X轴维度分布")
    axs[2].set_xlabel("长度")
    axs[2].set_ylabel("文件数量")
    axs[2].legend()
    
    # 5) 布局及保存
    plt.tight_layout()
    plt.savefig('output.png')
    plt.show()
    print("图像已保存为 output.png")


if __name__ == "__main__":
    # 修改为你的本地 nii.gz 文件夹路径
    folder_path = "/home/vipuser/Desktop/Data/Task02_PASp62_o/labelsTr"
    main(folder_path)




# import numpy as np
# import nibabel as nib

# t = np.load('/home/vipuser/Desktop/nnUNet/nnUNet_preprocessed/Dataset006_PASp61/nnUNetPlans_3d_fullres/501330_seg.npy')
# print(t.shape)
# image_nifti = nib.load('/home/vipuser/Desktop/Data/Task02_PASp62/imagesTr/501330.nii.gz')
# image_data = image_nifti.get_fdata()
# print(image_data.shape)
