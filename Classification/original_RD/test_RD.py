import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.calibration import CalibrationDisplay
import torch # 虽然导入了，但在提供的代码片段中未使用
import scipy.stats as st
import warnings
from sklearn.utils import resample # 需要导入 resample 用于 Bootstrap
import scipy.stats as st # 用于计算 DeLong CI (备选，这里主要用 Bootstrap)

warnings.filterwarnings('ignore')

# 配置日志 (保持不变)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test')



def calculate_auc_ci(y_true, y_pred_proba, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    使用 Bootstrap 计算 AUC 的置信区间。

    参数:
        y_true (np.ndarray): 真实标签 (一维数组)。
        y_pred_proba (np.ndarray): 模型预测的正类概率 (一维数组)。
        n_bootstraps (int): Bootstrap 重采样次数。
        alpha (float): 显著性水平 (例如 0.05 对应 95% CI)。
        seed (int): 随机种子，用于可复现的 Bootstrap 抽样。

    返回:
        tuple: (auc_point_estimate, auc_lower, auc_upper)
               如果无法计算（如数据不足），则可能返回 (NaN, NaN, NaN)。
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("AUC CI 计算跳过：真实标签少于两个类别。")
        return np.nan, np.nan, np.nan

    if len(y_true) != len(y_pred_proba):
        logger.error("AUC CI 计算错误：y_true 和 y_pred_proba 长度不匹配。")
        # 根据情况可以抛出异常或返回 NaN
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(seed) # 设置随机种子发生器
    bootstrapped_aucs = []
    n_samples = len(y_true)

    logger.debug(f"开始计算 AUC CI (Bootstrap, n={n_bootstraps})...") # 添加调试日志

    for i in range(n_bootstraps):
        # 1. 生成 Bootstrap 样本索引 (有放回抽样)
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            # 如果抽样后的样本只有一个类别，无法计算AUC，跳过此次迭代
            logger.debug(f"  Bootstrap 样本 {i+1} 跳过 (只有一个类别)。")
            continue

        # 2. 计算当前 Bootstrap 样本的 AUC
        try:
            fpr, tpr, _ = roc_curve(y_true[indices], y_pred_proba[indices])
            current_auc = auc(fpr, tpr)
            bootstrapped_aucs.append(current_auc)
        except Exception as e:
            # 捕获 roc_curve 或 auc 可能出现的其他罕见错误
            logger.warning(f"  计算 Bootstrap 样本 {i+1} 的 AUC 时出错: {e}")
            continue # 跳过此次迭代


    if not bootstrapped_aucs: # 如果所有 bootstrap 都失败了
         logger.error("AUC CI 计算失败：未能成功计算任何 Bootstrap 样本的 AUC。")
         # 计算原始AUC作为点估计，但CI为NaN
         try:
              fpr_orig, tpr_orig, _ = roc_curve(y_true, y_pred_proba)
              auc_point_estimate = auc(fpr_orig, tpr_orig)
         except:
              auc_point_estimate = np.nan
         return auc_point_estimate, np.nan, np.nan

    # 3. 计算点估计 (在原始数据上)
    try:
        fpr_orig, tpr_orig, _ = roc_curve(y_true, y_pred_proba)
        auc_point_estimate = auc(fpr_orig, tpr_orig)
    except Exception as e:
         logger.error(f"计算原始 AUC 点估计时出错: {e}")
         auc_point_estimate = np.nan # 如果原始AUC也失败

    # 4. 计算置信区间 (从 Bootstrap AUC 分布的百分位数)
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1 - alpha / 2.0) * 100
    auc_lower = np.percentile(bootstrapped_aucs, lower_percentile)
    auc_upper = np.percentile(bootstrapped_aucs, upper_percentile)

    logger.debug(f"AUC CI 计算完成: Point={auc_point_estimate:.4f}, Lower={auc_lower:.4f}, Upper={auc_upper:.4f}")

    return auc_point_estimate, auc_lower, auc_upper


def load_model(model_path, scaler_path, threshold_path=None):
    """
    加载训练好的模型、标准化器和最佳阈值
    
    参数:
        model_path: 模型文件路径
        scaler_path: 标准化器文件路径
        threshold_path: 阈值文件路径（可选）
        
    返回:
        model: 训练好的模型
        scaler: 标准化器
        threshold: 最佳阈值（如果提供了路径）
    """
    logger.info(f"加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    logger.info(f"加载标准化器: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    threshold = 0.5  # 默认阈值
    if threshold_path and os.path.exists(threshold_path):
        logger.info(f"加载最佳阈值: {threshold_path}")
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
    else:
        logger.warning(f"未找到阈值文件，使用默认阈值0.5 - {threshold_path}") # 增加路径显示
        
    return model, scaler, threshold

def load_data(features_path, labels_path):
    """
    加载和准备测试数据
    
    参数:
        features_path: 特征数据的路径
        labels_path: 标签数据的路径
        
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    logger.info(f"加载测试数据 - 特征: {features_path}, 标签: {labels_path}")
    
    # 加载数据
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    # 合并特征和标签
    df = pd.merge(features_df, labels_df, on='ID')
    
    # 提取特征和标签
    X = df.drop(['ID', 'label'], axis=1)
    y = df['label']
    
    logger.info(f"测试数据加载完成 - 样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    logger.info(f"标签分布: {y.value_counts().to_dict()}")
    
    return X, y

def calculate_youden_index(fpr, tpr, thresholds):
    """
    计算最佳截断点（Youden指数）
    
    参数:
        fpr: 假阳性率
        tpr: 真阳性率
        thresholds: 阈值
        
    返回:
        best_threshold: 最佳阈值
        best_youden: 最大Youden指数值
    """
    # 尤登指数 = 敏感性 + 特异性 - 1 = TPR - FPR
    youden_index = tpr - fpr
    max_youden_idx = np.argmax(youden_index)
    best_threshold = thresholds[max_youden_idx]
    best_youden = youden_index[max_youden_idx]
    
    return best_threshold, best_youden

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    计算性能指标
    
    参数:
        y_true: 真实标签
        y_pred_proba: 预测概率
        threshold: 决策阈值，默认0.5
        
    返回:
        metrics: 包含各种性能指标的字典
    """
    # 使用阈值将概率转换为预测标签
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 计算混淆矩阵
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError: # 处理只有一个类别被预测的情况
        logger.warning(f"在阈值 {threshold} 下，预测结果只包含一个类别。混淆矩阵可能不完整。")
        # 根据情况设置 tn, fp, fn, tp，例如全预测为0或全预测为1
        if np.all(y_pred == 0):
            tn = np.sum(y_true == 0)
            fp = 0
            fn = np.sum(y_true == 1)
            tp = 0
        elif np.all(y_pred == 1):
            tn = 0
            fp = np.sum(y_true == 0)
            fn = 0
            tp = np.sum(y_true == 1)
        else: # 其他罕见情况
             tn, fp, fn, tp = 0, 0, 0, 0

    # 计算各种指标
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性/召回率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0  # F1分数
    
    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
        'Threshold': threshold,
        'TP': tp, # 添加原始计数可能有用
        'FN': fn,
        'FP': fp,
        'TN': tn
    }
    
    return metrics

class DelongTest:
    """
    实现DeLong测试比较两个ROC曲线
    """
    def __init__(self, preds1, preds2, label, threshold=0.05):
        """
        初始化DeLong测试
        
        参数:
            preds1: 第一个模型的预测概率
            preds2: 第二个模型的预测概率
            label: 真实标签
            threshold: 显著性水平（默认0.05）
        """
        # 输入检查
        if len(preds1) != len(label) or len(preds2) != len(label):
             raise ValueError("预测概率和标签的长度必须一致")
        if len(np.unique(label)) != 2:
             raise ValueError("标签必须包含两个类别")
             
        self._preds1 = np.array(preds1)
        self._preds2 = np.array(preds2)
        self._label = np.array(label)
        self.threshold = threshold
        # 分离正负样本的预测值
        self._X_A, self._Y_A = self._group_preds_by_label(self._preds1, self._label)
        self._X_B, self._Y_B = self._group_preds_by_label(self._preds2, self._label)
        
        # 确保每个组至少有一个样本以避免计算错误
        if not self._X_A or not self._Y_A or not self._X_B or not self._Y_B:
            logger.warning("DeLong Test: 至少一个模型在一个类别上的预测为空，无法进行比较。")
            self.result = {
                'AUC_1': np.nan, 'AUC_2': np.nan, 'delta_AUC': np.nan,
                'z_score': np.nan, 'p_value': np.nan, 'significant': False,
                'error': 'Insufficient samples in one class'
            }
        else:
            self._show_result() # 只有在数据有效时才计算

    def _auc(self, X, Y) -> float:
        if not X or not Y: return np.nan # 处理空列表
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        """Mann-Whitney统计量"""
        return .5 if Y == X else float(Y < X) # 确保返回float

    def _structural_components(self, X, Y) -> tuple:
        if not X or not Y: return [], [] # 处理空列表
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        if len(V_A) < 2: return np.nan # 需要至少2个样本来计算方差
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        # 检查方差和协方差是否为 NaN
        if np.isnan(var_A) or np.isnan(var_B) or np.isnan(covar_AB):
            return np.nan
        # 检查分母是否接近于零或负数（可能由于样本量小导致方差估计不准）
        denominator_squared = var_A + var_B - 2 * covar_AB
        if denominator_squared <= 1e-10: # 阈值防止开根号负数或除以极小值
             logger.warning(f"DeLong Test: 方差项 ({var_A:.4f} + {var_B:.4f} - 2*{covar_AB:.4f} = {denominator_squared:.4f}) 过小或为负，无法计算有效的Z分数。")
             return np.nan
        return (auc_A - auc_B) / (denominator_squared**(.5)) # 移除+1e-8，直接检查分母

    def _group_preds_by_label(self, preds, actual) -> tuple:
        X = [p for (p, a) in zip(preds, actual) if a == 1] # 假设 1 是正类
        Y = [p for (p, a) in zip(preds, actual) if a == 0] # 假设 0 是负类
        return X, Y

    def _compute_z_p(self):
        V_A10, V_A01 = self._structural_components(self._X_A, self._Y_A)
        V_B10, V_B01 = self._structural_components(self._X_B, self._Y_B)

        auc_A = self._auc(self._X_A, self._Y_A)
        auc_B = self._auc(self._X_B, self._Y_B)

        # 检查AUC计算是否成功
        if np.isnan(auc_A) or np.isnan(auc_B):
            logger.warning("DeLong Test: AUC计算失败 (可能因为样本不足).")
            return np.nan, np.nan, auc_A, auc_B

        # 计算协方差矩阵S的条目
        len_V_A10, len_V_A01 = len(V_A10), len(V_A01)
        len_V_B10, len_V_B01 = len(V_B10), len(V_B01)

        # 检查样本数量是否足够计算方差/协方差
        if len_V_A10 < 2 or len_V_A01 < 2 or len_V_B10 < 2 or len_V_B01 < 2:
            logger.warning("DeLong Test: 样本量不足 (<2) 无法计算方差/协方差。")
            return np.nan, np.nan, auc_A, auc_B

        S_A1010 = self._get_S_entry(V_A10, V_A10, auc_A, auc_A)
        S_A0101 = self._get_S_entry(V_A01, V_A01, auc_A, auc_A)
        S_B1010 = self._get_S_entry(V_B10, V_B10, auc_B, auc_B)
        S_B0101 = self._get_S_entry(V_B01, V_B01, auc_B, auc_B)
        S_A10B10 = self._get_S_entry(V_A10, V_B10, auc_A, auc_B)
        S_A01B01 = self._get_S_entry(V_A01, V_B01, auc_A, auc_B)

        # 检查 S entries 是否有效
        if any(np.isnan(s) for s in [S_A1010, S_A0101, S_B1010, S_B0101, S_A10B10, S_A01B01]):
            logger.warning("DeLong Test: 方差/协方差条目计算失败。")
            return np.nan, np.nan, auc_A, auc_B

        var_A = S_A1010 / len_V_A10 + S_A0101 / len_V_A01
        var_B = S_B1010 / len_V_B10 + S_B0101 / len_V_B01
        covar_AB = S_A10B10 / len_V_A10 + S_A01B01 / len_V_A01

        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)

        # 只有在z分数有效时计算p值
        if np.isnan(z):
            p = np.nan
        else:
            # 双尾检验
            p = st.norm.sf(abs(z)) * 2

        return z, p, auc_A, auc_B

    def _show_result(self):
        z, p, auc_1, auc_2 = self._compute_z_p()
        # 即使 p 是 NaN，significant 也应该是 False
        significant = False
        if not np.isnan(p):
            significant = p < self.threshold

        self.result = {
            'AUC_1': auc_1,
            'AUC_2': auc_2,
            'delta_AUC': auc_1 - auc_2 if not (np.isnan(auc_1) or np.isnan(auc_2)) else np.nan,
            'z_score': z,
            'p_value': p,
            'significant': significant # 使用计算出的 significant
        }

        # 避免在日志中打印 NaN
        auc_diff_str = f"{self.result['delta_AUC']:.4f}" if not np.isnan(self.result['delta_AUC']) else "N/A"
        z_str = f"{z:.4f}" if not np.isnan(z) else "N/A"
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"

        logger.info(f"DeLong测试结果:")
        logger.info(f"  AUC差异: {auc_diff_str}")
        logger.info(f"  Z分数: {z_str}")
        logger.info(f"  P值: {p_str}")
        if np.isnan(p):
             logger.info("  结论: 无法确定显著性 (计算错误或样本不足)")
        elif self.result['significant']:
            logger.info(f"  结论: 两个ROC曲线在 alpha={self.threshold} 水平下有显著差异")
        else:
            logger.info(f"  结论: 两个ROC曲线在 alpha={self.threshold} 水平下无显著差异")

def decision_curve_analysis(y_true, y_pred_proba, threshold_range=np.arange(0.01, 1.0, 0.01)):
    """
    计算决策曲线分析的净效益 (修正版)
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
    prevalence = event_count / N

    # 计算模型净收益
    model_net_benefit = []
    for pt in threshold_range: 
        # 找出所有预测概率 >= pt 的样本
        treat_mask = y_pred_proba >= pt 
        
        n_treat = np.sum(treat_mask) # 多少人被模型建议治疗
        if n_treat == 0: # 如果没有样本达到这个阈值，净收益无法计算（或认为是0？）
             # 通常，如果没人被治疗，则净收益为0，因为没有获益也没有损失
             # 但如果严格按公式 (tp/N - fp/N * odds)，当tp=fp=0时也为0
             net_ben = 0.0
             model_net_benefit.append(net_ben)
             continue

        # 在这些被选择治疗的样本中计算 TP 和 FP
        tp = np.sum(y_true[treat_mask] == 1)
        fp = np.sum(y_true[treat_mask] == 0)
        
        # 处理 pt 接近 1 导致分母为0的情况
        if abs(1 - pt) < 1e-8:
            # 当 pt=1 时，理论上净收益是负无穷，除非模型完美区分
            # 但实践中，如果 tp > 0 且 fp = 0，可以认为收益是 tp/N
            # 如果 fp > 0，则收益是负无穷
            # 为了绘图，通常避免 pt=1 或设为一个非常小的值
            # 另一种处理：如果pt=1，直接设为None/NaN，让绘图库忽略
            net_ben = np.nan # 或者根据具体情况设定
        else:
            # 净效益公式: (TP / N) - (FP / N) * (pt / (1 - pt))
            net_ben = (tp / N) - (fp / N) * (pt / (1 - pt))
            
        model_net_benefit.append(net_ben)

    # 计算'全部治疗'策略的净效益
    all_net_benefit = []
    for pt in threshold_range:
        if abs(1 - pt) < 1e-8:
            all_net_ben = np.nan # 避免在 pt=1 时计算
        else:
            # Net Benefit (Treat All) = Prevalence - (1 - Prevalence) * (pt / (1 - pt))
            all_net_ben = prevalence - (1 - prevalence) * (pt / (1 - pt))
        all_net_benefit.append(all_net_ben)
            
    # 计算'无人治疗'策略的净效益（总是为0）
    none_net_benefit = np.zeros_like(threshold_range)
    
    dca_results = {
        'thresholds': threshold_range,
        'model_net_benefit': np.array(model_net_benefit), # 转为numpy array方便后续处理
        'all_net_benefit': np.array(all_net_benefit),
        'none_net_benefit': none_net_benefit
    }
    
    return dca_results

# --- plot_roc_curves, plot_calibration_curves, plot_dca_curves 函数保持不变 ---
def plot_roc_curves(model_names, y_true, y_pred_probas, auc_results, save_path=None): # 添加 auc_results 参数
    """
    绘制多个模型的ROC曲线，并在图例中包含AUC (95% CI)。
    """
    plt.figure(figsize=(10, 8))

    fprs = []
    tprs = []

    for i, model_name in enumerate(model_names):
        y_prob = y_pred_probas[i]
        auc_res = auc_results.get(model_name, {'point': np.nan, 'lower': np.nan, 'upper': np.nan}) # 安全获取

        if y_prob is None or len(y_prob) == 0 or np.isnan(auc_res['point']):
             logger.warning(f"模型 {model_name} 的预测概率无效或AUC计算失败，跳过ROC绘制。")
             label = f'{model_name} (AUC = N/A)'
             plt.plot([0, 1], [0, 1], '--', label=label) # 画一条虚线表示错误/缺失
             fprs.append(np.array([0, 1]))
             tprs.append(np.array([0, 1]))
             continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fprs.append(fpr)
        tprs.append(tpr)

        # --- 修改图例标签 ---
        if not np.isnan(auc_res['lower']) and not np.isnan(auc_res['upper']):
             label = f"{model_name} (AUC = {auc_res['point']:.3f} [{auc_res['lower']:.3f}-{auc_res['upper']:.3f}])"
        else:
             label = f"{model_name} (AUC = {auc_res['point']:.3f} [CI N/A])" # CI 无法计算的情况
        # --- ----------- ---

        plt.plot(fpr, tpr, lw=2, label=label) # 使用更新后的标签

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic (ROC) Curves with 95% CI') # 更新标题
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存ROC曲线失败: {e}")

    plt.show()

    # 返回原始AUC点估计列表（如果需要）
    return [auc_results.get(name, {}).get('point', np.nan) for name in model_names], fprs, tprs

def plot_calibration_curves(model_names, y_true, y_pred_probas, save_path=None, n_bins=10): # 增加 n_bins 到 10
    """
    绘制多个模型的校准曲线
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8)) # 创建 figure 和 axes
    
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated") # 添加完美校准线

    for i, model_name in enumerate(model_names):
        if y_pred_probas[i] is None or len(y_pred_probas[i]) == 0:
             logger.warning(f"模型 {model_name} 的预测概率为空或无效，跳过校准曲线绘制。")
             continue
             
        try:
             # 使用 from_estimator 或 from_predictions
             # from_predictions 需要 y_true, y_prob
             display = CalibrationDisplay.from_predictions(
                 y_true,
                 y_pred_probas[i],
                 n_bins=n_bins,
                 name=model_name,
                 ax=ax # 指定绘制在哪个 axes 上
             )
        except ValueError as e:
             logger.error(f"绘制模型 {model_name} 的校准曲线时出错: {e}")
             logger.error(f"可能是因为预测概率值超出 [0, 1] 范围或全是相同值。")
             # 可以选择跳过这个模型，或者绘制一个标记表示错误
             ax.text(0.1, 0.9 - i*0.05, f"{model_name}: Error plotting calibration", color='red', transform=ax.transAxes)


    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title('Calibration plots (Reliability Curves)')
    ax.grid(True, alpha=0.3)

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"校准曲线已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存校准曲线失败: {e}")

    plt.show()

def plot_dca_curves(model_names, dca_results_list, save_path=None):
    """
    绘制多个模型的决策曲线 (修正绘图样式和错误处理)
    """
    plt.figure(figsize=(10, 8))
    
    min_benefit = 0 # 用于动态调整Y轴下限
    max_benefit_overall = -np.inf # 用于动态调整Y轴上限

    # 绘制每个模型的曲线
    for i, model_name in enumerate(model_names):
        dca_result = dca_results_list[i]
        thresholds = dca_result['thresholds']
        model_benefits = dca_result['model_net_benefit']

        # 过滤掉无效值以便绘图和计算范围
        valid_indices = np.isfinite(model_benefits)
        if not np.any(valid_indices): # 如果全是 NaN
             logger.warning(f"模型 {model_name} 的DCA净收益全部无效，无法绘制。")
             plt.plot([], [], label=f'{model_name} (Error computing benefit)', linestyle=':') # 画个空线占位
             continue

        valid_thresholds = thresholds[valid_indices]
        valid_benefits = model_benefits[valid_indices]

        plt.plot(valid_thresholds, valid_benefits, label=f'{model_name}', lw=2)
        min_benefit = min(min_benefit, np.min(valid_benefits)) # 更新Y轴下限
        max_benefit_overall = max(max_benefit_overall, np.max(valid_benefits)) # 更新Y轴上限

    # 绘制 'Treat All' 和 'Treat None' 曲线 (只需要第一个模型的结果中的参考曲线)
    if dca_results_list: # 确保列表不为空
        ref_dca = dca_results_list[0]
        thresholds = ref_dca['thresholds']
        all_benefits = ref_dca['all_net_benefit']
        none_benefits = ref_dca['none_net_benefit'] # 应该是全零

        # Treat All
        valid_indices_all = np.isfinite(all_benefits)
        if np.any(valid_indices_all):
            valid_thresholds_all = thresholds[valid_indices_all]
            valid_benefits_all = all_benefits[valid_indices_all]
            plt.plot(valid_thresholds_all, valid_benefits_all,
                     'k-', label='Treat All', lw=1.5, linestyle='--') # 黑色虚线
            min_benefit = min(min_benefit, np.min(valid_benefits_all))
            max_benefit_overall = max(max_benefit_overall, np.max(valid_benefits_all))

        # Treat None (y=0)
        plt.plot(thresholds, none_benefits,
                 'k:', label='Treat None', lw=1.5) # 黑色点线

    # 设置坐标轴范围和标签
    plt.xlim([0.0, 1.0])
    # 动态调整Y轴，给一点边距
    y_lower = min(min_benefit, 0) - 0.05 # 确保至少包含0，并留边距
    y_upper = max(max_benefit_overall, 0) + 0.05 # 确保至少包含0，并留边距
    if y_lower < -0.5: y_lower = -0.5 # 限制Y轴负向不要太夸张
    if np.isinf(y_upper): y_upper = 0.5 # 如果最大值是inf，设置一个默认上限
    plt.ylim([y_lower, y_upper])

    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis (DCA)')

    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"DCA曲线已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存DCA曲线失败: {e}")

    plt.show()


# ================== 修改后的 test_models 函数 ==================
def test_models(models_info, X_test, y_test, results_dir=None, n_bootstraps_auc=1000):
    """
    测试多个模型并比较它们的性能，计算并显示AUC的95% CI，将DeLong检验结果保存到单独的CSV。

    Args:
        models_info (dict): 包含模型信息的字典，键是模型名称，值是包含 'model', 'scaler', 'threshold' 的字典。
        X_test (pd.DataFrame or np.ndarray): 测试集特征。
        y_test (pd.Series or np.ndarray): 测试集真实标签。
        results_dir (str, optional): 保存结果的目录。 Defaults to None.
        n_bootstraps_auc (int, optional): 用于计算AUC CI的Bootstrap次数。Defaults to 1000.

    Returns:
        dict: 包含测试结果的字典，例如:
              {'metrics_summary': DataFrame,
               'delong_comparison_summary': DataFrame,
               'auc_results': dict,
               'y_pred_probas': list,
               'thresholds_used': list,
               'dca_results': list}
    """
    model_names = list(models_info.keys())
    logger.info(f"开始评估模型: {model_names}")

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    y_pred_probas_positive = [] # 存储正类的1D概率
    y_pred_labels = []
    metrics_list = []
    thresholds_used = [] # 存储每个模型实际使用的阈值
    dca_results_list = []
    model_errors = {} # 记录模型加载或预测中的错误
    # === 新增：存储AUC及其CI ===
    auc_results = {} # 存储格式: {model_name: {'point': auc, 'lower': ci_low, 'upper': ci_up}}
    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test # Ensure numpy array

    # 1. 评估每个模型
    for model_name in model_names:
        logger.info(f"评估模型: {model_name}")
        try:
            model = models_info[model_name]['model']
            scaler = models_info[model_name]['scaler']
            threshold = models_info[model_name]['threshold'] # 使用预定义的阈值
            thresholds_used.append(threshold) # 记录使用的阈值

            # 确保 X_test 是 DataFrame 或 Ndarray
            if isinstance(X_test, pd.DataFrame):
                 X_test_scaled = scaler.transform(X_test)
            elif isinstance(X_test, np.ndarray):
                 X_test_scaled = scaler.transform(X_test)
            else:
                 raise TypeError(f"X_test 类型不受支持: {type(X_test)}")


            # 获取预测概率 (通常是两列)
            y_pred_proba_all = model.predict_proba(X_test_scaled)

            # --- 选择正类（通常是索引1）的概率 ---
            if y_pred_proba_all.ndim == 2 and y_pred_proba_all.shape[1] >= 2:
                 # 假设正类是第二列 (索引 1)
                 y_pred_proba = y_pred_proba_all[:, 1]
            elif y_pred_proba_all.ndim == 1: # 处理可能直接返回1D数组的模型
                 y_pred_proba = y_pred_proba_all
            else:
                 logger.error(f"模型 {model_name} 的 predict_proba 输出格式不符合预期: {y_pred_proba_all.shape}")
                 raise ValueError(f"模型 {model_name} 输出概率格式错误") # 抛出异常

            # 检查概率值是否在 [0, 1] 范围内
            if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
                logger.warning(f"模型 {model_name} 的预测概率包含超出 [0, 1] 范围的值。将进行截断。")
                y_pred_proba = np.clip(y_pred_proba, 0, 1) # 截断到 [0, 1]

            y_pred_probas_positive.append(y_pred_proba) # 存储1D数组

            # --- 计算AUC及其95% CI ---
            # 确保 y_test_np 在这里被使用
            auc_point, auc_lower, auc_upper = calculate_auc_ci(
                y_test_np, # 使用转换后的 numpy array
                y_pred_proba,
                n_bootstraps=n_bootstraps_auc, # 使用函数参数
                seed=42 # 固定随机种子以保证复现性
            )
            auc_results[model_name] = {'point': auc_point, 'lower': auc_lower, 'upper': auc_upper}
            auc_ci_str = f"{auc_point:.3f} ({auc_lower:.3f}-{auc_upper:.3f})" if not np.isnan(auc_point) else "N/A"
            # --- -------------------- ---

            # 基于阈值计算预测标签 (使用1D概率)
            y_pred = (y_pred_proba >= threshold).astype(int)
            y_pred_labels.append(y_pred)

            # 计算性能指标 (使用1D概率数组)
            metrics = calculate_metrics(y_test_np, y_pred_proba, threshold=threshold) # 使用 numpy array
            metrics_list.append(metrics)

            # 决策曲线分析 (使用1D概率数组)
            dca_result = decision_curve_analysis(y_test_np, y_pred_proba) # 使用 numpy array
            dca_results_list.append(dca_result)

            # 更新日志输出，包含CI
            logger.info(f"  模型: {model_name}, AUC (95% CI): {auc_ci_str}, 使用阈值: {threshold:.4f}")
            logger.info(f"  准确率: {metrics['Accuracy']:.4f}, 敏感性: {metrics['Sensitivity']:.4f}, 特异性: {metrics['Specificity']:.4f}")
            model_errors[model_name] = None # 标记该模型成功处理

        except Exception as e:
             logger.error(f"评估模型 {model_name} 时发生错误: {e}", exc_info=False) # 可以设置 exc_info=True 查看完整堆栈
             # 添加占位符结果，以便后续处理不会因缺少元素而出错
             y_pred_probas_positive.append(None) # 使用None标记错误
             y_pred_labels.append(None)
             metrics_list.append({k: np.nan for k in ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'Threshold', 'TP', 'FN', 'FP', 'TN']})
             thresholds_used.append(np.nan)
             dca_results_list.append({'thresholds': np.arange(0.01, 1.0, 0.01),
                                      'model_net_benefit': np.full(99, np.nan),
                                      'all_net_benefit': np.full(99, np.nan),
                                      'none_net_benefit': np.zeros(99)})
             model_errors[model_name] = str(e) # 记录错误信息
             # === 也要为失败的模型添加空的AUC结果 ===
             auc_results[model_name] = {'point': np.nan, 'lower': np.nan, 'upper': np.nan}


    # 筛选出成功处理的模型和它们的概率，用于比较
    valid_indices = [i for i, prob in enumerate(y_pred_probas_positive) if prob is not None]
    valid_model_names = [model_names[i] for i in valid_indices]
    valid_y_pred_probas = [y_pred_probas_positive[i] for i in valid_indices]
    valid_y_pred_labels = [y_pred_labels[i] for i in valid_indices] # 也筛选标签，用于相似性检查

    # 2. 检查预测结果相似性 (只比较成功处理的模型)
    logger.info("检查有效模型预测结果相似性:")
    if len(valid_model_names) > 1:
        for i in range(len(valid_model_names)):
            for j in range(i + 1, len(valid_model_names)):
                model1 = valid_model_names[i]
                model2 = valid_model_names[j]
                try:
                    # 确保比较的是非None的标签数组
                    if valid_y_pred_labels[i] is not None and valid_y_pred_labels[j] is not None:
                        same_predictions = np.mean(valid_y_pred_labels[i] == valid_y_pred_labels[j])
                        logger.info(f"  {model1} 和 {model2} 的预测标签相同比例: {same_predictions:.4f}")
                        if same_predictions > 0.99:
                            logger.warning(f"  注意: {model1} 和 {model2} 的预测标签几乎完全相同")
                    else:
                         logger.warning(f"无法比较 {model1} 和 {model2} 的标签相似性，因为其中一个或两个的标签未能生成。")
                except Exception as e:
                    logger.error(f"比较 {model1} 和 {model2} 相似性时出错: {e}")
    else:
        logger.info("  只有一个或没有有效模型，无法进行相似性比较。")


    # 3. 模型性能比较 (DeLong Test - 只比较成功处理的模型)
    logger.info("模型性能比较 (DeLong Test):")
    delong_results_list = [] # 初始化存储 DeLong 结果的列表
    if len(valid_model_names) > 1:
        for i in range(len(valid_model_names)):
            for j in range(i + 1, len(valid_model_names)):
                model1 = valid_model_names[i]
                model2 = valid_model_names[j]
                logger.info(f"  比较: {model1} vs {model2}")
                try:
                    # 使用存储在 valid_y_pred_probas 中的1D概率数组和 numpy 标签数组
                    delong_test = DelongTest(valid_y_pred_probas[i], valid_y_pred_probas[j], y_test_np)
                    result = delong_test.result

                    # 将结果存储到列表中
                    delong_result_row = {
                        'Model_1': model1,
                        'Model_2': model2,
                        'AUC_1': result['AUC_1'],
                        'AUC_2': result['AUC_2'],
                        'delta_AUC': result['delta_AUC'],
                        'z_score': result['z_score'],
                        'p_value': result['p_value'],
                        'Significant (p<0.05)': result['significant'] # 使用更清晰的列名
                    }
                    delong_results_list.append(delong_result_row)

                    # 日志输出结果
                    if 'error' in result and result['error']:
                         logger.warning(f"    DeLong测试跳过，原因: {result['error']}")
                    elif np.isnan(result['p_value']):
                         logger.warning(f"    DeLong测试P值无法计算 (可能是样本不足或方差问题)")
                    elif result['significant']:
                        better_model = model1 if result['delta_AUC'] > 0 else model2
                        worse_model = model2 if better_model == model1 else model1
                        logger.info(f"    {better_model} 显著优于 {worse_model} (p={result['p_value']:.4f})")
                    else:
                        logger.info(f"    {model1} 和 {model2} 性能无显著差异 (p={result['p_value']:.4f})")

                except Exception as e:
                     logger.error(f"    运行 DeLong 测试 ({model1} vs {model2}) 时出错: {e}")
                     # 记录一个失败的比较结果
                     delong_results_list.append({
                        'Model_1': model1, 'Model_2': model2, 'AUC_1': np.nan, 'AUC_2': np.nan,
                        'delta_AUC': np.nan, 'z_score': np.nan, 'p_value': np.nan,
                        'Significant (p<0.05)': False, 'Error': str(e) # 添加错误列
                    })

    else:
        logger.info("  只有一个或没有有效模型，无法进行 DeLong 比较。")

    # 4. 绘图 (使用所有模型的概率，绘图函数内部会处理None)
    if results_dir:
        roc_save_path = os.path.join(results_dir, "roc_curves.png")
        calibration_save_path = os.path.join(results_dir, "calibration_curves.png")
        dca_save_path = os.path.join(results_dir, "dca_curves.png")
    else:
        roc_save_path, calibration_save_path, dca_save_path = None, None, None

    # 传递 auc_results 给 plot_roc_curves
    # plot_roc_curves 函数需要被修改以接受和使用 auc_results (如上一个回答所示)
    plot_roc_curves(model_names, y_test_np, y_pred_probas_positive, auc_results, save_path=roc_save_path)
    plot_calibration_curves(model_names, y_test_np, y_pred_probas_positive, save_path=calibration_save_path)
    plot_dca_curves(model_names, dca_results_list, save_path=dca_save_path)

    # 5. 汇总和保存结果
    # (a) 单个模型的性能指标 (包含格式化的 AUC CI)
    metrics_summary_list = []
    for i, model_name in enumerate(model_names):
        auc_res = auc_results.get(model_name, {'point': np.nan, 'lower': np.nan, 'upper': np.nan}) # 安全获取
        # 格式化 AUC (95% CI) 字符串
        if not np.isnan(auc_res['point']):
            auc_ci_str = f"{auc_res['point']:.3f} ({auc_res['lower']:.3f}-{auc_res['upper']:.3f})"
        else:
            auc_ci_str = "N/A" # 或者保持 NaN

        # 从 metrics_list 安全获取指标，处理可能的NaN
        current_metrics = metrics_list[i] if i < len(metrics_list) else {}

        metrics_summary_list.append({
            'Model': model_name,
            'AUC (95% CI)': auc_ci_str, # 使用格式化字符串列
            # 使用 .get() 提供默认值 NaN，防止 current_metrics 不完整时出错
            'Accuracy': current_metrics.get('Accuracy', np.nan),
            'Sensitivity': current_metrics.get('Sensitivity', np.nan),
            'Specificity': current_metrics.get('Specificity', np.nan),
            'PPV': current_metrics.get('PPV', np.nan),
            'NPV': current_metrics.get('NPV', np.nan),
            'F1': current_metrics.get('F1', np.nan),
            'Threshold_Used': thresholds_used[i] if i < len(thresholds_used) else np.nan, # 安全获取阈值
            'Error': model_errors.get(model_name, None) # 添加错误信息列
        })
    metrics_summary = pd.DataFrame(metrics_summary_list)

    # (b) DeLong 检验的两两比较结果
    delong_summary_df = pd.DataFrame(delong_results_list) # 从列表创建 DataFrame

    # 保存文件
    if results_dir:
        metrics_summary_path = os.path.join(results_dir, "performance_metrics.csv")
        delong_summary_path = os.path.join(results_dir, "model_comparison_delong.csv") # <--- 新文件名

        try:
            metrics_summary.to_csv(metrics_summary_path, index=False, float_format='%.4f')
            logger.info(f"性能指标汇总表已保存至: {metrics_summary_path}")
        except Exception as e:
            logger.error(f"保存性能指标汇总表失败: {e}")

        if not delong_summary_df.empty: # 只有在有结果时才保存
            try:
                delong_summary_df.to_csv(delong_summary_path, index=False, float_format='%.4f')
                logger.info(f"DeLong检验比较结果已保存至: {delong_summary_path}")
            except Exception as e:
                logger.error(f"保存DeLong检验结果失败: {e}")

    # 打印到控制台
    print("\n--- 测试集模型性能指标汇总 ---")
    print(metrics_summary.to_string(index=False, float_format='%.4f')) # float_format 对字符串列 AUC (95% CI) 无效

    if not delong_summary_df.empty:
        print("\n--- 测试集模型两两比较 (DeLong Test) 结果 ---")
        print(delong_summary_df.to_string(index=False, float_format='%.4f'))
    else:
        print("\n--- 未进行模型间的 DeLong Test 比较 (少于2个有效模型) ---")

    # 返回结果
    return {
        'metrics_summary': metrics_summary,
        'delong_comparison_summary': delong_summary_df,
        'auc_results': auc_results, # 返回详细的AUC点估计和CI
        'y_pred_probas': y_pred_probas_positive, # 返回1D概率列表 (可能包含None)
        'thresholds_used': thresholds_used, # 返回使用的阈值列表
        'dca_results': dca_results_list # 返回DCA结果列表
    }

# ================== main 函数保持不变 ==================
def main():
    """主函数"""
    # 配置
    config = {
        'test_features_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/RD/final_test_features.csv',
        'test_labels_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_labels.csv',
        'results_dir': '/home/vipuser/Desktop/Classification/test_results_RD',
        'models_dir': '/home/vipuser/Desktop/Classification/train_results_RD',  # 训练结果目录
        'models_to_test': ['LR', 'SVM', 'ANN']
    }

    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)

    # 加载测试数据
    X_test, y_test = load_data(config['test_features_path'], config['test_labels_path'])

    # 加载模型和标准化器
    models_info = {}
    for model_type in config['models_to_test']:
        model_dir = os.path.join(config['models_dir'], model_type)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_type.lower()}_scaler.pkl")
        threshold_path = os.path.join(model_dir, f"{model_type.lower()}_threshold.pkl") # 小写扩展名

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model, scaler, threshold = load_model(model_path, scaler_path, threshold_path)
                models_info[model_type] = {
                    'model': model,
                    'scaler': scaler,
                    'threshold': threshold
                }
            except Exception as e:
                 logger.error(f"加载模型 {model_type} 的文件时出错 ({model_path} or {scaler_path}): {e}")
        else:
            logger.warning(f"模型 {model_type} 的文件不存在或不完整，跳过评估。检查路径:")
            logger.warning(f"  模型: {model_path} (存在: {os.path.exists(model_path)})")
            logger.warning(f"  标准化器: {scaler_path} (存在: {os.path.exists(scaler_path)})")


    # 测试模型
    test_results = None # 初始化
    if models_info:
        test_results = test_models(models_info, X_test, y_test, results_dir=config['results_dir'])
        logger.info("模型测试完成")
    else:
        logger.error("没有成功加载任何有效的模型，无法进行测试")

    # 你可以在这里使用 test_results 字典中的内容，比如访问:
    # metrics_df = test_results['metrics_summary']
    # delong_df = test_results['delong_comparison_summary']

    return test_results # 返回包含所有结果的字典

if __name__ == "__main__":
    main()