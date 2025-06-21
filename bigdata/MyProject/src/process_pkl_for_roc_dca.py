import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import logging
import scipy.stats as st
import warnings

# --- 配置日志 (保持不变) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def decision_curve_analysis(y_true, y_pred_proba, threshold_range=np.arange(0.01, 1.0, 0.01)):
    """
    计算决策曲线分析的净效益 (来自你的代码)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    N = len(y_true)
    if N == 0:
        logger.warning("DCA: 输入数据为空。")
        return {'thresholds': threshold_range,
                'model_net_benefit': np.full_like(threshold_range, np.nan),
                'all_net_benefit': np.full_like(threshold_range, np.nan),
                'none_net_benefit': np.zeros_like(threshold_range)}

    event_count = np.sum(y_true)
    # Handle case where N might be zero, though checked above
    prevalence = event_count / N if N > 0 else 0

    # 计算模型净收益
    model_net_benefit = []
    for pt in threshold_range:
        treat_mask = y_pred_proba >= pt
        n_treat = np.sum(treat_mask)

        if n_treat == 0:
            net_ben = 0.0
            model_net_benefit.append(net_ben)
            continue

        tp = np.sum(y_true[treat_mask] == 1)
        fp = np.sum(y_true[treat_mask] == 0)

        if abs(1 - pt) < 1e-8:
            net_ben = np.nan
        else:
             # Check N to prevent division by zero if N was 0 initially
            if N > 0:
                net_ben = (tp / N) - (fp / N) * (pt / (1 - pt))
            else:
                net_ben = np.nan # Should not happen due to earlier check

        model_net_benefit.append(net_ben)

    # 计算'全部治疗'策略的净效益
    all_net_benefit = []
    for pt in threshold_range:
        if abs(1 - pt) < 1e-8:
            all_net_ben = np.nan
        else:
            # Check prevalence calculation validity
            if N > 0:
                all_net_ben = prevalence - (1 - prevalence) * (pt / (1 - pt))
            else:
                all_net_ben = np.nan
        all_net_benefit.append(all_net_ben)

    # 计算'无人治疗'策略的净效益
    none_net_benefit = np.zeros_like(threshold_range)

    dca_results = {
        'thresholds': threshold_range,
        'model_net_benefit': np.array(model_net_benefit),
        'all_net_benefit': np.array(all_net_benefit),
        'none_net_benefit': none_net_benefit
    }

    return dca_results


def plot_roc_curves(model_names, y_true, y_pred_probas, save_path=None):
    """
    绘制多个模型的ROC曲线 - 增强字体版本
    """
    # 保存原始设置
    original_rcParams = plt.rcParams.copy()
    
    # 设置超大字体
    plt.rcParams.update({
        'font.size': 28,              # 基础字体
        'axes.titlesize': 30,         # 标题
        'axes.labelsize': 28,         # 轴标签
        'xtick.labelsize': 22,        # x轴刻度
        'ytick.labelsize': 22,        # y轴刻度
        'legend.fontsize': 26,        # 图例
        'figure.titlesize': 30        # 图标题
    })
    
    plt.figure(figsize=(12, 10))  # 增大图形尺寸以配合大字体
    aucs = []
    fprs = []
    tprs = []

    # --- 修改：使用更丰富的颜色循环 ---
    colors = plt.cm.get_cmap('tab10', len(model_names)) # 使用 tab10 色彩映射

    for i, model_name in enumerate(model_names):
        if y_pred_probas[i] is None or len(y_pred_probas[i]) == 0:
             logger.warning(f"模型 {model_name} 的预测概率为空或无效，跳过ROC绘制。")
             aucs.append(np.nan)
             fprs.append(np.array([0, 1]))
             tprs.append(np.array([0, 1]))
             plt.plot([0, 1], [0, 1], '--', label=f'{model_name} (AUC = N/A - Error)', 
                     color=colors(i), linewidth=3) # 增加线宽
             continue

        # Check if y_true and y_pred_probas[i] have enough samples and classes
        try:
            if len(np.unique(y_true)) < 2:
                 logger.error(f"绘制 ROC 曲线错误: 真实标签 y_true 只有一个类别。")
                 raise ValueError("y_true must contain at least two classes.")
            if len(y_true) != len(y_pred_probas[i]):
                 logger.error(f"绘制 ROC 曲线错误: y_true ({len(y_true)}) 和 {model_name} 概率 ({len(y_pred_probas[i])}) 长度不匹配。")
                 raise ValueError("Input lengths differ.")

            fpr, tpr, _ = roc_curve(y_true, y_pred_probas[i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            fprs.append(fpr)
            tprs.append(tpr)
            # --- 修改：为每条曲线指定颜色，增加线宽 ---
            plt.plot(fpr, tpr, lw=4, label=f'{model_name} (AUC = {roc_auc:.3f})', 
                    color=colors(i))  # 线宽从2增加到4
        except Exception as e:
            logger.error(f"计算或绘制模型 {model_name} 的ROC时出错: {e}")
            aucs.append(np.nan)
            fprs.append(np.array([0, 1]))
            tprs.append(np.array([0, 1]))
            plt.plot([0, 1], [0, 1], '--', label=f'{model_name} (AUC = N/A - Error: {e})', 
                    color=colors(i), linewidth=3) # 增加线宽


    plt.plot([0, 1], [0, 1], 'k--', lw=3, label='Chance (AUC = 0.500)')  # 线宽从2增加到3
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)', fontweight='bold', labelpad=10)
    plt.ylabel('Sensitivity (True Positive Rate)', fontweight='bold', labelpad=10)
    plt.title('ROC Curves Comparison', 
              fontweight='bold', pad=20)
    
    # 调整图例位置和样式
    plt.legend(loc="lower right", frameon=True, fancybox=True, 
               shadow=True, framealpha=0.9, edgecolor='black')
    
    # 增强网格线
    plt.grid(True, alpha=0.4, linewidth=1.5)
    
    # 加粗坐标轴
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 设置刻度线宽度
    ax.tick_params(width=2, length=8)
    
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高DPI到600
            # 同时保存PDF版本
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"ROC曲线已保存至: {save_path} 和 {pdf_path}")
        except Exception as e:
            logger.error(f"保存ROC曲线失败: {e}")

    plt.close() # 关闭图形
    
    # 恢复原始设置
    plt.rcParams.update(original_rcParams)
    
    return aucs, fprs, tprs


def plot_dca_curves(model_names, dca_results_list, save_path=None):
    """
    绘制多个模型的决策曲线 - 增强字体版本
    """
    # 保存原始设置
    original_rcParams = plt.rcParams.copy()
    
    # 设置超大字体
    plt.rcParams.update({
        'font.size': 28,              # 基础字体
        'axes.titlesize': 30,         # 标题
        'axes.labelsize': 28,         # 轴标签
        'xtick.labelsize': 22,        # x轴刻度
        'ytick.labelsize': 22,        # y轴刻度
        'legend.fontsize': 26,        # 图例
        'figure.titlesize': 30        # 图标题
    })
    
    plt.figure(figsize=(12, 10))  # 增大图形尺寸

    min_benefit = 0
    max_benefit_overall = -np.inf

    # --- 修改：使用更丰富的颜色循环 ---
    colors = plt.cm.get_cmap('tab10', len(model_names)) # 使用 tab10 色彩映射

    # 绘制每个模型的曲线
    for i, model_name in enumerate(model_names):
        if i >= len(dca_results_list) or not isinstance(dca_results_list[i], dict):
             logger.warning(f"缺少模型 {model_name} 的有效 DCA 结果，跳过绘制。")
             plt.plot([], [], label=f'{model_name} (Missing DCA data)', 
                     linestyle=':', linewidth=3)
             continue

        dca_result = dca_results_list[i]
        thresholds = dca_result.get('thresholds')
        model_benefits = dca_result.get('model_net_benefit')

        if thresholds is None or model_benefits is None:
            logger.warning(f"模型 {model_name} 的 DCA 结果字典缺少键或值为 None，跳过绘制。")
            plt.plot([], [], label=f'{model_name} (Invalid DCA data)', 
                    linestyle=':', linewidth=3)
            continue

        valid_indices = np.isfinite(model_benefits)
        if not np.any(valid_indices):
             logger.warning(f"模型 {model_name} 的DCA净收益全部无效，无法绘制。")
             plt.plot([], [], label=f'{model_name} (Error computing benefit)', 
                     linestyle=':', linewidth=3)
             continue

        valid_thresholds = thresholds[valid_indices]
        valid_benefits = model_benefits[valid_indices]

        # --- 修改：为每条曲线指定颜色，增加线宽 ---
        plt.plot(valid_thresholds, valid_benefits, label=f'{model_name}', 
                lw=4, color=colors(i))  # 线宽从2增加到4
        min_benefit = min(min_benefit, np.min(valid_benefits))
        max_benefit_overall = max(max_benefit_overall, np.max(valid_benefits))

    # 绘制 'Treat All' 和 'Treat None' 曲线
    if dca_results_list and isinstance(dca_results_list[0], dict):
        ref_dca = dca_results_list[0]
        thresholds = ref_dca.get('thresholds')
        all_benefits = ref_dca.get('all_net_benefit')
        none_benefits = ref_dca.get('none_net_benefit')

        if thresholds is not None and all_benefits is not None and none_benefits is not None:
            valid_indices_all = np.isfinite(all_benefits)
            if np.any(valid_indices_all):
                valid_thresholds_all = thresholds[valid_indices_all]
                valid_benefits_all = all_benefits[valid_indices_all]
                plt.plot(valid_thresholds_all, valid_benefits_all,
                         'k-', label='Treat All', lw=3, linestyle='--')  # 线宽增加
                min_benefit = min(min_benefit, np.min(valid_benefits_all))
                max_benefit_overall = max(max_benefit_overall, np.max(valid_benefits_all))
            else:
                 logger.warning("无法绘制 'Treat All' DCA 曲线，净收益无效。")
            plt.plot(thresholds, none_benefits,
                     'k:', label='Treat None', lw=3)  # 线宽增加
        else:
            logger.warning("无法绘制 DCA 参考线，数据不完整。")

    # 设置坐标轴范围和标签
    plt.xlim([0.0, 1.0])
    y_lower = 0 - 0.05
    y_upper = max(max_benefit_overall, 0) + 0.05
    if y_lower < -0.5: y_lower = -0.5
    if np.isinf(y_upper) or np.isnan(y_upper): y_upper = 0.6
    plt.ylim([y_lower, y_upper])

    plt.xlabel('Threshold Probability', fontweight='bold', labelpad=10)
    plt.ylabel('Net Benefit', fontweight='bold', labelpad=10)
    plt.title('DCA Comparison', 
              fontweight='bold', pad=20)
    
    # 调整图例
    plt.legend(loc="upper right", frameon=True, fancybox=True, 
               shadow=True, framealpha=0.9, edgecolor='black')
    
    # 增强网格线
    plt.grid(True, alpha=0.4, linewidth=1.5)
    
    # 加粗坐标轴
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 设置刻度线宽度
    ax.tick_params(width=2, length=8)
    
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 提高DPI到600
            # 同时保存PDF版本
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"DCA曲线已保存至: {save_path} 和 {pdf_path}")
        except Exception as e:
            logger.error(f"保存DCA曲线失败: {e}")

    plt.close() # 关闭图形
    
    # 恢复原始设置
    plt.rcParams.update(original_rcParams)


# =============================================================================
# ===== 主脚本流程 (保持不变) =================================================
# =============================================================================

# --- 1. 配置区域 ---
pkl_file_paths = [
    # '/home/vipuser/Desktop/bigdata/MyProject/results/resnet18_20250402_044228/test_results/ensemble_probs.pkl', # 示例 1
    "/home/vipuser/Desktop/test_new_file/edge/resnet18_20250402_044228/test_results_3/ensemble_probs.pkl",   # 替换路径 1
    "/home/vipuser/Desktop/test_new_file/edge/resnet18_20250403_013532_SingleBranch/test_results_3/ensemble_probs.pkl",  # 替换路径 2
    "/home/vipuser/Desktop/test_new_file/global/resnet18_20250404_084156/test_results_3/ensemble_probs.pkl",  # 替换路径 4
    "/home/vipuser/Desktop/test_new_file/global/resnet18_20250404_055529_SingleBranch/test_results_3/ensemble_probs.pkl",   # 替换路径 3
    "/home/vipuser/Desktop/test_new_file/resnet18_20250404_192331/test_results_3/ensemble_probs.pkl",   # 替换路径 5
]
# pkl_file_paths = [
#     # '/home/vipuser/Desktop/bigdata/MyProject/results/resnet18_20250402_044228/test_results/ensemble_probs.pkl', # 示例 1
#     "/home/vipuser/Desktop/test_new_file1/edge/resnet18_20250423_134608/test_results_3/ensemble_probs.pkl",   # 替换路径 1
#     "/home/vipuser/Desktop/test_new_file1/edge/resnet18_20250422_150738_SingleBranch/test_results_3/ensemble_probs.pkl",  # 替换路径 2
#     "/home/vipuser/Desktop/test_new_file1/global/resnet18_20250424_012017/test_results_3/ensemble_probs.pkl",  # 替换路径 4
#     "/home/vipuser/Desktop/test_new_file1/global/resnet18_20250424_062753_SingleBranch/test_results_3/ensemble_probs.pkl",   # 替换路径 3
#     "/home/vipuser/Desktop/test_new_file1/resnet18_20250424_092825/test_results_3/ensemble_probs.pkl",   # 替换路径 5
# ]
# pkl_file_paths = [
#     # '/home/vipuser/Desktop/bigdata/MyProject/results/resnet18_20250402_044228/test_results/ensemble_probs.pkl', # 示例 1
#     "/home/vipuser/Desktop/test_new_file2/edge/resnet18_20250424_170121/test_results_3/ensemble_probs.pkl",   # 替换路径 1
#     "/home/vipuser/Desktop/test_new_file2/edge/resnet18_20250424_212947_SingleBranch/test_results_3/ensemble_probs.pkl",  # 替换路径 2
#     "/home/vipuser/Desktop/test_new_file2/global/resnet18_20250425_152119/test_results_3/ensemble_probs.pkl",  # 替换路径 4
#     "/home/vipuser/Desktop/test_new_file2/global/resnet18_20250425_224506_SingleBranch/test_results_3/ensemble_probs.pkl",   # 替换路径 3
#     "/home/vipuser/Desktop/test_new_file2/resnet18_20250426_023454/test_results_3/ensemble_probs.pkl",   # 替换路径 5
# ]
# --- 修改：为每个文件生成模型名称 ---
# 你可以手动指定，或者从路径中提取部分信息
individual_model_names = [
    "Local-Dual",    # 强调分支和双输入特性
    "Local-Single",  # 基线对比：局部单输入
    "Global-Dual",   # 强调分支和双输入特性
    "Global-Single", # 基线对比：全局单输入
    "Dual-Branch",      # 明确指出这是最终的融合模型
]
# individual_model_names = [
#     "L-Dual",    # Local Branch, Dual-Input
#     "L-Single",  # Local Branch, Single-Input
#     "G-Dual",    # Global Branch, Dual-Input
#     "G-Single",  # Global Branch, Single-Input
#     "Dual-Branch",   # 你的最终方案
# ]

# 确保 model_names 数量与 pkl_file_paths 数量一致
if len(individual_model_names) != len(pkl_file_paths):
    logger.error("错误：individual_model_names 列表的数量与 pkl_file_paths 列表的数量不匹配！")
    exit()

results_dir = "/home/vipuser/Desktop/bigdata/MyProject/results"

# --- 确保结果目录存在 ---
try:
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"结果将保存到: {results_dir}")
except OSError as e:
    logger.error(f"创建目录 {results_dir} 时出错: {e}")
    exit()

# --- 2. 加载概率和标签，并检查一致性 ---
all_probs_list = []
all_labels_list = []
logger.info("开始加载 .pkl 文件...")

for i, file_path in enumerate(pkl_file_paths):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if 'ensemble_probs' not in data or 'true_labels' not in data:
                logger.error(f"文件 {file_path} 缺少 'ensemble_probs' 或 'true_labels' 键。")
                exit()
            probs = data['ensemble_probs']
            labels = data['true_labels']
            if isinstance(probs, list): probs = np.array(probs)
            if isinstance(labels, list): labels = np.array(labels)

            if probs.ndim > 1:
                 if probs.shape[1] == 2: probs = probs[:, 1]
                 elif probs.shape[1] == 1: probs = probs.flatten()
                 else: raise ValueError(f"概率维度不明确: {probs.shape}")
            elif probs.ndim == 0: raise ValueError("概率不是数组")

            if labels.ndim > 1:
                 if labels.shape[1] == 1: labels = labels.flatten()
                 else: raise ValueError(f"标签维度不明确: {labels.shape}")
            elif labels.ndim == 0: raise ValueError("标签不是数组")

            logger.info(f"成功加载: {os.path.basename(file_path)} (作为 '{individual_model_names[i]}'), 样本数: {len(probs)}")
            if len(probs) != len(labels):
                 logger.error(f"文件 {file_path} 概率 ({len(probs)}) 与标签 ({len(labels)}) 数量不匹配！")
                 exit()
            all_probs_list.append(probs)
            all_labels_list.append(labels)
    except FileNotFoundError:
        logger.error(f"文件未找到 {file_path}")
        exit()
    except Exception as e:
        logger.error(f"加载或处理文件 {file_path} 时出错: {e}")
        exit()

logger.info(f"已加载 {len(all_probs_list)} 个模型的数据。")
if not all_probs_list:
    logger.error("未能成功加载任何概率数据。")
    exit()

# --- 检查一致性 ---
first_len = len(all_probs_list[0])
if not all(len(p) == first_len for p in all_probs_list):
    logger.error("并非所有文件的样本数量都相同！")
    exit()
logger.info("正在检查所有文件中的 'true_labels' 是否一致...")
y_true = all_labels_list[0] # 使用第一个文件的标签作为基准
for i in range(1, len(all_labels_list)):
    if not np.array_equal(y_true, all_labels_list[i]):
        logger.error(f"文件 {pkl_file_paths[i]} (模型 '{individual_model_names[i]}') 的 'true_labels' 与第一个文件不一致！")
        exit()
logger.info("'true_labels' 在所有文件中一致。使用此标签作为 Ground Truth。")
logger.info(f"总样本数: {len(y_true)}")

# --- 3. 生成并保存 ROC 曲线 ---
logger.info("生成并保存各模型 ROC 曲线对比图...")
roc_filename = "roc_curves_comparison.png"
roc_save_path = os.path.join(results_dir, roc_filename)
plot_roc_curves(individual_model_names, y_true, all_probs_list, save_path=roc_save_path)

# --- 4. 生成并保存 DCA 曲线 ---
logger.info("生成并保存各模型 DCA 曲线对比图...")
try:
    dca_results_list_for_plotting = []
    for i, prob_list in enumerate(all_probs_list):
        model_name = individual_model_names[i]
        logger.debug(f"为模型 '{model_name}' 计算 DCA...")
        if prob_list is None or len(prob_list) == 0:
             logger.warning(f"模型 '{model_name}' 的概率数据无效，无法计算 DCA。")
             dca_results_list_for_plotting.append({
                'thresholds': np.arange(0.01, 1.0, 0.01),
                'model_net_benefit': np.full(99, np.nan),
                'all_net_benefit': np.full(99, np.nan),
                'none_net_benefit': np.zeros(99),
                'error': 'Invalid input probability'
             })
             continue

        dca_result_dict = decision_curve_analysis(y_true, prob_list)
        dca_results_list_for_plotting.append(dca_result_dict)

    # 如果列表中至少有一个有效的DCA结果，用它的参考线数据填充可能失败的模型的参考线
    first_valid_dca = next((d for d in dca_results_list_for_plotting if 'error' not in d), None)
    if first_valid_dca:
        ref_all_benefit = first_valid_dca.get('all_net_benefit', np.full(99, np.nan))
        ref_none_benefit = first_valid_dca.get('none_net_benefit', np.zeros(99))
        ref_thresholds = first_valid_dca.get('thresholds', np.arange(0.01, 1.0, 0.01))
        for dca_dict in dca_results_list_for_plotting:
            if 'error' in dca_dict:
                dca_dict['all_net_benefit'] = ref_all_benefit
                dca_dict['none_net_benefit'] = ref_none_benefit
                dca_dict['thresholds'] = ref_thresholds

    dca_filename = "dca_curves_comparison.png"
    dca_save_path = os.path.join(results_dir, dca_filename)
    plot_dca_curves(individual_model_names, dca_results_list_for_plotting, save_path=dca_save_path)

except Exception as e:
    logger.error(f"\n生成或保存 DCA 曲线时出错: {e}")

# --- 5. 完成提示 ---
logger.info("\n脚本执行完毕。对比图已尝试保存。")