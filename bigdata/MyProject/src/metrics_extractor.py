import pandas as pd
import re
import os
import numpy as np
from pathlib import Path

# 创建一个函数来解析指标文件
def parse_metrics_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式提取指标
        acc_match = re.search(r'准确率: (\d+\.\d+)%', content)
        auc_match = re.search(r'AUC: (\d+\.\d+)', content)
        ci_match = re.search(r'95% CI: ([\d\.\-]+)', content)
        f1_match = re.search(r'F1分数: (\d+\.\d+)', content)
        sen_match = re.search(r'敏感度: (\d+\.\d+)', content)
        spe_match = re.search(r'特异度: (\d+\.\d+)', content)
        
        # 提取数值
        acc = float(acc_match.group(1))/100 if acc_match else None
        auc = float(auc_match.group(1)) if auc_match else None
        ci = ci_match.group(1) if ci_match else None
        f1 = float(f1_match.group(1)) if f1_match else None
        sen = float(sen_match.group(1)) if sen_match else None
        spe = float(spe_match.group(1)) if spe_match else None
        
        return {
            'ACC': acc,
            'AUC': auc,
            'CI': ci,
            'F1': f1,
            'SEN': sen,
            'SPE': spe
        }
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return None

# 定义文件夹路径和模型信息
# 请用实际的文件夹路径和模型信息替换这些示例
# folders_info = [
#     {"path": "/home/vipuser/Desktop/test_new_file/edge/resnet18_20250403_013532_SingleBranch", "网络结构": "ResNet18单输入", "input": "局部胎盘图像"},
#     {"path": "/home/vipuser/Desktop/test_new_file/edge/resnet18_20250402_044228/", "网络结构": "ResNet18双输入", "input": "局部胎盘图像+边界带"},
#     {"path": "/home/vipuser/Desktop/test_new_file/global/resnet18_20250404_055529_SingleBranch", "网络结构": "ResNet18单输入", "input": "完整图像"},
#     {"path": "/home/vipuser/Desktop/test_new_file/global/resnet18_20250404_084156", "网络结构": "ResNet18双输入", "input": "完整图像+胎盘区域"},
#     {"path": "/home/vipuser/Desktop/test_new_file/resnet18_20250404_192331", "网络结构": "双分支模型", "input": ""}
# ]
# folders_info = [
#     {"path": "/home/vipuser/Desktop/test_new_file1/edge/resnet18_20250422_150738_SingleBranch", "网络结构": "ResNet34单输入", "input": "局部胎盘图像"},
#     {"path": "/home/vipuser/Desktop/test_new_file1/edge/resnet18_20250423_134608", "网络结构": "ResNet34双输入", "input": "局部胎盘图像+边界带"},
#     {"path": "/home/vipuser/Desktop/test_new_file1/global/resnet18_20250424_062753_SingleBranch", "网络结构": "ResNet34单输入", "input": "完整图像"},
#     {"path": "/home/vipuser/Desktop/test_new_file1/global/resnet18_20250424_012017", "网络结构": "ResNet34双输入", "input": "完整图像+胎盘区域"},
#     {"path": "/home/vipuser/Desktop/test_new_file1/resnet18_20250424_092825", "网络结构": "双分支模型", "input": ""}
# ]
folders_info = [
    {"path": "/home/vipuser/Desktop/test_new_file2/edge/resnet18_20250424_212947_SingleBranch", "网络结构": "ResNet50单输入", "input": "局部胎盘图像"},
    {"path": "/home/vipuser/Desktop/test_new_file2/edge/resnet18_20250424_170121", "网络结构": "ResNet50双输入", "input": "局部胎盘图像+边界带"},
    {"path": "/home/vipuser/Desktop/test_new_file2/global/resnet18_20250425_224506_SingleBranch", "网络结构": "ResNet50单输入", "input": "完整图像"},
    {"path": "/home/vipuser/Desktop/test_new_file2/global/resnet18_20250425_152119", "网络结构": "ResNet50双输入", "input": "完整图像+胎盘区域"},
    {"path": "/home/vipuser/Desktop/test_new_file2/resnet18_20250426_023454", "网络结构": "双分支模型", "input": ""}
]
# 初始化结果列表
results = []

# 处理每个文件夹
for info in folders_info:
    folder = info["path"]
    file_path = os.path.join(folder, "test_results", "ensemble_youden_metrics.txt")
    
    # 解析文件获取指标
    if os.path.exists(file_path):
        metrics = parse_metrics_file(file_path)
        if metrics:
            metrics.update({
                'folder': folder,
                '网络结构': info["网络结构"],
                'input': info["input"]
            })
            results.append(metrics)
        else:
            print(f"无法解析文件: {file_path}")
    else:
        print(f"文件不存在: {file_path}")
        # 使用占位符数据进行演示
        metrics = {
            'folder': folder,
            '网络结构': info["网络结构"],
            'input': info["input"],
            'ACC': np.nan,
            'AUC': np.nan,
            'CI': 'N/A',
            'F1': np.nan,
            'SEN': np.nan,
            'SPE': np.nan
        }
        results.append(metrics)

# 创建DataFrame
df = pd.DataFrame(results)

# 格式化列，添加CI到AUC
df['AUC (95% CI)'] = df.apply(
    lambda row: f"{row['AUC']:.3f} ({row['CI']})" if not pd.isna(row['AUC']) else "N/A", 
    axis=1
)

# 创建最终表格格式
final_table = pd.DataFrame({
    '网络结构': df['网络结构'],
    '输入图像': df['input'],
    # '训练集 AUC (95% CI)': df['AUC (95% CI)'],  # 假设训练集和测试集AUC相同，根据实际情况调整
    '测试集 AUC (95% CI)': df['AUC (95% CI)'],
    '测试集 ACC': df['ACC'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"),
    '测试集 SEN': df['SEN'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"),
    '测试集 SPE': df['SPE'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"),
    'F1 score': df['F1'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"),
})

# 找出每一列的最大值并加粗（在控制台输出中用*表示）
max_indices = {}
for col in ['测试集 ACC', '测试集 SEN', '测试集 SPE', 'F1 score']:
    # 移除N/A并转换为数值以找出最大值
    values = pd.to_numeric(final_table[col].replace('N/A', np.nan), errors='coerce')
    if not values.isna().all():  # 确保有非NA值
        max_idx = values.idxmax()
        max_indices[col] = max_idx

# 打印表格（在控制台中高亮最大值）
print("\n医学影像分类指标汇总表:\n")
print(final_table.to_string(index=False))
print("\n* 每列中的最大值标记如下:")
for col, idx in max_indices.items():
    print(f"  {col}: {final_table.loc[idx, col]} (行 {idx+1})")

# 保存为CSV文件
graph_path = "/home/vipuser/Desktop/bigdata/MyProject/results/医学影像分类指标汇总表.csv"
final_table.to_csv(graph_path, index=False, encoding='utf-8-sig')
print("\n表格已保存为 'medical_imaging_metrics_summary.csv'")

# 创建加粗最大值的HTML表格
def bold_max(val, col, max_indices):
    if col in max_indices and final_table.loc[max_indices[col], col] == val:
        return f'<b>{val}</b>'
    return val

html_table = final_table.copy()
for col in max_indices.keys():
    html_table[col] = html_table.apply(
        lambda row: bold_max(row[col], col, max_indices), 
        axis=1
    )

# # 保存为HTML文件，以便在浏览器中查看带有加粗最大值的表格
# html = """
# <!DOCTYPE html>
# <html>
# <head>
#     <meta charset="UTF-8">
#     <title>医学影像分类指标汇总表</title>
#     <style>
#         table { border-collapse: collapse; width: 100%; }
#         th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
#         th { background-color: #f2f2f2; }
#         tr:nth-child(even) { background-color: #f9f9f9; }
#     </style>
# </head>
# <body>
#     <h2>医学影像分类指标汇总表</h2>
# """ + html_table.to_html(index=False, escape=False) + """
# </body>
# </html>
# """

# html_path = "/home/vipuser/Desktop/bigdata/MyProject/results/医学影像分类指标汇总表.html"
# with open(html_path, "w", encoding="utf-8") as f:
#     f.write(html)

# print("带有格式化的HTML表格已保存为 '医学影像分类指标汇总表.html'")