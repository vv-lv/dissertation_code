import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.calibration import CalibrationDisplay
import scipy.stats as st # 用于 DeLong 测试的 p 值计算
import warnings


# 配置日志 (保持不变)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test') # 日志记录器叫 'test'

def calculate_auc_ci(y_true, y_pred_proba, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    用 Bootstrap 方法计算 AUC 的置信区间。

    参数:
        y_true (np.ndarray): 真实标签 (一维)。
        y_pred_proba (np.ndarray): 模型预测为正类的概率 (一维)。
        n_bootstraps (int): Bootstrap 重复采样的次数。
        alpha (float): 显著性水平 (比如 0.05 代表 95% 置信区间)。
        seed (int): 随机种子，让 Bootstrap 采样结果可复现。

    返回:
        tuple: (auc点估计值, 置信区间下限, 置信区间上限)
               如果算不了 (比如数据不够)，可能返回 (NaN, NaN, NaN)。
    """

    # 检查标签和预测概率长度是否一致
    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true 和 y_pred_proba 的长度不一致。")

    rng = np.random.RandomState(seed) # 创建一个独立的随机数生成器
    bootstrapped_aucs = [] # 存每次 Bootstrap 抽样的 AUC 值
    n_samples = len(y_true)

    logger.debug(f"开始计算 AUC 置信区间 (Bootstrap 法, 重复次数 n={n_bootstraps})...")

    for i in range(n_bootstraps):
        # 1. 生成 Bootstrap 样本的索引 (有放回地抽样)
        indices = rng.randint(0, n_samples, n_samples)

        # 2. 计算当前 Bootstrap 样本的 AUC
        # 注意：保留这里的 try-except 是为了处理单次 bootstrap 失败不影响整体的情况
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred_proba[indices])
        current_auc = auc(fpr, tpr)
        bootstrapped_aucs.append(current_auc)

    # 3. 计算点估计值 (用原始的、未经抽样的数据算一次 AUC)
    fpr_orig, tpr_orig, _ = roc_curve(y_true, y_pred_proba)
    auc_point_estimate = auc(fpr_orig, tpr_orig)

    # 4. 计算置信区间 (取 Bootstrap 得到的 AUC 分布的百分位数)
    lower_percentile = (alpha / 2.0) * 100 # 比如 alpha=0.05 -> 2.5
    upper_percentile = (1 - alpha / 2.0) * 100 # 比如 alpha=0.05 -> 97.5
    auc_lower = np.percentile(bootstrapped_aucs, lower_percentile)
    auc_upper = np.percentile(bootstrapped_aucs, upper_percentile)

    logger.debug(f"AUC CI 计算完成: 点估计={auc_point_estimate:.4f}, 下限={auc_lower:.4f}, 上限={auc_upper:.4f}")

    return auc_point_estimate, auc_lower, auc_upper

def load_model(model_path, scaler_path, threshold_path=None):
    """
    加载之前训练好的模型、标准化器（Scaler）和推荐阈值。

    参数:
        model_path (str): 模型文件 (.pkl) 的路径。
        scaler_path (str): Scaler 文件 (.pkl) 的路径。
        threshold_path (str, 可选): 阈值文件 (.pkl) 的路径。

    返回:
        tuple: (模型对象, Scaler对象, 阈值)
               如果没提供阈值路径或文件不存在，阈值会是默认的 0.5。
    """
    logger.info(f"加载模型文件: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"加载标准化器文件: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    threshold = 0.5  # 设置一个默认阈值
    if threshold_path and os.path.exists(threshold_path):
        logger.info(f"加载推荐阈值文件: {threshold_path}")
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
    elif threshold_path:
        # 如果提供了路径但文件不存在
        raise FileNotFoundError(f"推荐阈值文件不存在: {threshold_path}")

    return model, scaler, threshold

def load_data(features_path, labels_path):
    """
    加载测试用的特征和标签数据。

    参数:
        features_path (str): 特征 CSV 文件路径。
        labels_path (str): 标签 CSV 文件路径。

    返回:
        tuple: (X, y) X 是特征 DataFrame，y 是标签 Series。
    """
    logger.info(f"加载测试数据 - 特征: {features_path}, 标签: {labels_path}")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # 用 'ID' 列来合并特征和标签
    df = pd.merge(features_df, labels_df, on='ID')

    feature_cols = [col for col in df.columns if col not in ['ID', 'label']]
    X = df[feature_cols]
    y = df['label']

    logger.info(f"测试数据加载完成 - 样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    logger.info(f"测试集标签分布: {y.value_counts().to_dict()}") # 显示各类别的数量

    return X, y

# 必要时计算最佳阈值
def calculate_youden_index(fpr, tpr, thresholds):
    """
    根据 ROC 曲线计算最佳阈值 (基于约登指数)。

    参数:
        fpr (np.ndarray): 假正例率数组。
        tpr (np.ndarray): 真正例率数组。
        thresholds (np.ndarray): 对应的阈值数组。

    返回:
        tuple: (最佳阈值, 最大约登指数值)
    """
    # 约登指数 = 敏感性 + 特异性 - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    youden_index = tpr - fpr
    max_youden_idx = np.argmax(youden_index) # 找到最大指数的位置
    best_threshold = thresholds[max_youden_idx]
    best_youden = youden_index[max_youden_idx]

    # 这里可以加一个检查 best_threshold 是否 inf 的处理，但在这个脚本里没用到
    return best_threshold, best_youden

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    根据真实标签、预测概率和指定阈值计算性能指标。

    参数:
        y_true (np.ndarray): 真实标签 (0 或 1)。
        y_pred_proba (np.ndarray): 模型预测为正类 (1) 的概率。
        threshold (float, 可选): 分类阈值，默认 0.5。

    返回:
        dict: 包含各种指标的字典。
    """
    # 根据阈值，把概率转换成预测的类别 (0 或 1)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 计算混淆矩阵 (TP, FP, FN, TN)
    # 保留这里的 try-except，因为预测结果可能全是0或全是1，导致 ravel() 失败
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


    # 计算各种指标，注意处理分母为零的情况
    accuracy = accuracy_score(y_true, y_pred) # 准确率
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性 (Sensitivity) / 召回率 (Recall) / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性 (Specificity) / TNR
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0          # 阳性预测值 (PPV) / 精确率 (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # 阴性预测值 (NPV)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0 # F1 分数

    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
        'Threshold': threshold, 
        'TP': tp, 
        'FN': fn,
        'FP': fp,
        'TN': tn
    }

    return metrics

class DelongTest:
    """
    用 DeLong 的方法比较两条 ROC 曲线是否有显著差异。
    代码是基于公开的实现修改而来。
    """
    def __init__(self, preds1, preds2, label, threshold=0.05):
        """
        初始化 DeLong 测试。

        参数:
            preds1 (np.ndarray): 第一个模型预测的正类概率。
            preds2 (np.ndarray): 第二个模型预测的正类概率。
            label (np.ndarray): 真实的标签 (0 或 1)。
            threshold (float): 显著性水平 (P值阈值)，默认 0.05。
        """
        self._preds1 = np.asarray(preds1)
        self._preds2 = np.asarray(preds2)
        self._label = np.asarray(label)
        self.threshold = threshold

        # 按标签把预测概率分组 (X组是正类预测值, Y组是负类预测值)
        self._X_A, self._Y_A = self._group_preds_by_label(self._preds1, self._label)
        self._X_B, self._Y_B = self._group_preds_by_label(self._preds2, self._label)

        self._calculate_and_store_result()

    def _auc(self, X, Y):
        """计算单个模型的 AUC (基于 Mann-Whitney 统计量)"""
        # AUC = P(X > Y) + 0.5 * P(X = Y)
        return 1/(len(X)*len(Y)) * sum(self._kernel(x, y) for x in X for y in Y)

    def _kernel(self, X, Y):
        """Mann-Whitney U 检验的核心部分"""
        # 如果 X > Y 返回 1, 如果 X == Y 返回 0.5, 如果 X < Y 返回 0
        return 0.5 if Y == X else float(Y < X) # 确保返回浮点数

    def _structural_components(self, X, Y):
        """计算 DeLong 测试需要的结构成分 V10 和 V01"""
        if not X or not Y: return [], [] # 处理空列表
        # V10[i] = P(Y < X[i]) + 0.5 * P(Y == X[i])
        V10 = [1/len(Y) * sum(self._kernel(x, y) for y in Y) for x in X]
        # V01[j] = P(X > Y[j]) + 0.5 * P(X == Y[j])
        V01 = [1/len(X) * sum(self._kernel(x, y) for x in X) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B):
        """计算协方差矩阵 S 的单个元素"""
        # S_entry = Cov(V_A, V_B)
        return 1/(len(V_A)-1) * sum((a - auc_A)*(b - auc_B) for a, b in zip(V_A, V_B))

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        """计算 Z 分数"""
        # 检查输入的方差和协方差是不是有效的数
        # 计算 Z 分数的分母的平方: Var(AUC_A - AUC_B) = Var(A) + Var(B) - 2*Covar(A, B)
        denominator_squared = var_A + var_B - 2 * covar_AB
        # 检查分母是不是太小或者负数 (可能因为样本少导致方差估计不稳)
        if denominator_squared <= 1e-10: # 用一个小阈值防止开根号负数或除以零
             logger.warning(f"DeLong 测试警告：方差项 ({var_A:.4f} + {var_B:.4f} - 2*{covar_AB:.4f} = {denominator_squared:.4f}) 太小或为负，无法计算有效的 Z 分数。")
             return np.nan
        # Z = (AUC_A - AUC_B) / sqrt(Var(AUC_A - AUC_B))
        return (auc_A - auc_B) / (denominator_squared**0.5)

    def _group_preds_by_label(self, preds, actual):
        """按真实标签把预测概率分组"""
        # 假设类别 1 是正类，类别 0 是负类
        X = [p for (p, a) in zip(preds, actual) if a == 1] # 正类样本的预测概率
        Y = [p for (p, a) in zip(preds, actual) if a == 0] # 负类样本的预测概率
        return X, Y

    def _calculate_and_store_result(self):
        """计算 DeLong 测试的 Z 分数和 P 值，并保存结果"""
        # 计算两个模型各自的结构成分
        V_A10, V_A01 = self._structural_components(self._X_A, self._Y_A)
        V_B10, V_B01 = self._structural_components(self._X_B, self._Y_B)

        # 计算两个模型的 AUC
        auc_A = self._auc(self._X_A, self._Y_A)
        auc_B = self._auc(self._X_B, self._Y_B)

        # 检查 AUC 是否成功计算出来
        if np.isnan(auc_A) or np.isnan(auc_B):
            logger.warning("DeLong 测试警告：AUC 计算失败 (可能是因为某个类别的样本不足)。")
            self.result = {'AUC_1': auc_A, 'AUC_2': auc_B, 'delta_AUC': np.nan, 'z_score': np.nan, 'p_value': np.nan, 'significant': False, 'error': 'AUC 计算失败'}
            return

        # 检查样本量是否足够计算方差/协方差
        len_V_A10, len_V_A01 = len(V_A10), len(V_A01)
        len_V_B10, len_V_B01 = len(V_B10), len(V_B01)
        if len_V_A10 < 2 or len_V_A01 < 2 or len_V_B10 < 2 or len_V_B01 < 2:
            logger.warning("DeLong 测试警告：某个类别的样本量不足 (<2)，无法计算方差/协方差。")
            self.result = {'AUC_1': auc_A, 'AUC_2': auc_B, 'delta_AUC': np.nan, 'z_score': np.nan, 'p_value': np.nan, 'significant': False, 'error': '样本量不足计算方差'}
            return

        # 计算协方差矩阵 S 的所有需要的元素
        S_A1010 = self._get_S_entry(V_A10, V_A10, auc_A, auc_A) # Var(V_A10)
        S_A0101 = self._get_S_entry(V_A01, V_A01, auc_A, auc_A) # Var(V_A01)
        S_B1010 = self._get_S_entry(V_B10, V_B10, auc_B, auc_B) # Var(V_B10)
        S_B0101 = self._get_S_entry(V_B01, V_B01, auc_B, auc_B) # Var(V_B01)
        S_A10B10 = self._get_S_entry(V_A10, V_B10, auc_A, auc_B) # Cov(V_A10, V_B10)
        S_A01B01 = self._get_S_entry(V_A01, V_B01, auc_A, auc_B) # Cov(V_A01, V_B01)

        # 检查这些方差/协方差计算是否成功
        if any(np.isnan(s) for s in [S_A1010, S_A0101, S_B1010, S_B0101, S_A10B10, S_A01B01]):
            logger.warning("DeLong 测试警告：方差/协方差矩阵元素计算失败。")
            self.result = {'AUC_1': auc_A, 'AUC_2': auc_B, 'delta_AUC': np.nan, 'z_score': np.nan, 'p_value': np.nan, 'significant': False, 'error': '方差/协方差计算失败'}
            return

        # 计算 AUC_A 和 AUC_B 的方差估计值
        var_A = S_A1010 / len_V_A10 + S_A0101 / len_V_A01
        var_B = S_B1010 / len_V_B10 + S_B0101 / len_V_B01
        # 计算 AUC_A 和 AUC_B 的协方差估计值
        covar_AB = S_A10B10 / len_V_A10 + S_A01B01 / len_V_A01

        # 计算 Z 分数
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)

        # 只有当 Z 分数有效时，才计算 P 值
        if np.isnan(z):
            p = np.nan
        else:
            # 计算双尾 P 值 (从标准正态分布的生存函数 sf)
            # sf(x) = 1 - cdf(x)
            p = st.norm.sf(abs(z)) * 2
        # 判断结果是否显著
        significant = False
        if not np.isnan(p): # 只有 P 值有效时才判断
            significant = p < self.threshold

        # 把所有结果存到 self.result 字典里
        self.result = {
            'AUC_1': auc_A,
            'AUC_2': auc_B,
            'delta_AUC': auc_A - auc_B, # AUC 差值
            'z_score': z,
            'p_value': p,
            'significant': significant
        }

        # 打印结果到日志 (避免打印 NaN)
        auc_diff_str = f"{self.result['delta_AUC']:.4f}" if not np.isnan(self.result['delta_AUC']) else "N/A"
        z_str = f"{z:.4f}" if not np.isnan(z) else "N/A"
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"

        logger.info(f"DeLong 测试结果:")
        logger.info(f"  AUC 差值 (AUC1 - AUC2): {auc_diff_str}")
        logger.info(f"  Z 分数: {z_str}")
        logger.info(f"  P 值: {p_str}")
        if np.isnan(p):
             logger.info("  结论: 无法确定显著性 (计算错误或样本不足)")
        elif self.result['significant']:
            logger.info(f"  结论: 两条 ROC 曲线在 alpha={self.threshold} 水平下有显著差异")
        else:
            logger.info(f"  结论: 两条 ROC 曲线在 alpha={self.threshold} 水平下无显著差异")


def decision_curve_analysis(y_true, y_pred_proba, threshold_range=np.arange(0.01, 1.0, 0.01)):
    """
    计算决策曲线分析 (DCA) 的净效益 (Net Benefit)。

    参数:
        y_true (np.ndarray): 真实标签 (0 或 1)。
        y_pred_proba (np.ndarray): 模型预测为正类 (1) 的概率。
        threshold_range (np.ndarray, 可选): 要评估的阈值概率范围，默认 0.01 到 0.99。

    返回:
        dict: 包含阈值和对应净效益的字典。
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    N = len(y_true) # 总样本量

    event_count = np.sum(y_true) # 阳性样本数量
    prevalence = event_count / N # 阳性样本比例 (患病率)

    # 计算模型在不同阈值下的净效益
    model_net_benefit = []
    for pt in threshold_range: # pt 是当前考虑的阈值概率
        # 找出模型预测概率 >= pt 的样本，假设这些样本被建议采取干预（治疗）
        treat_mask = y_pred_proba >= pt

        n_treat = np.sum(treat_mask) # 有多少样本被建议干预

        # 在被建议干预的样本中，计算真阳性(TP)和假阳性(FP)的数量
        tp = np.sum(y_true[treat_mask] == 1)
        fp = np.sum(y_true[treat_mask] == 0)

        # 处理阈值 pt 接近 1 的情况，避免除以零
        if abs(1 - pt) < 1e-8: # 如果 pt 几乎等于 1
             # 理论上 pt=1 时净效益可能是负无穷（如果FP>0）。为了绘图方便，设为 NaN。
             net_ben = np.nan
        else:
            # 净效益公式: (真阳性比例) - (假阳性比例) * (阈值概率的比值 odds)
            # Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
            net_ben = (tp / N) - (fp / N) * (pt / (1 - pt))

        model_net_benefit.append(net_ben)

    # 计算‘全部干预’策略的净效益
    all_net_benefit = []
    for pt in threshold_range:
        if abs(1 - pt) < 1e-8: # 同样处理 pt=1 的情况
            all_net_ben = np.nan
        else:
            # Net Benefit (Treat All) = Prevalence - (1 - Prevalence) * (pt / (1 - pt))
            all_net_ben = prevalence - (1 - prevalence) * (pt / (1 - pt))
        all_net_benefit.append(all_net_ben)

    # 计算‘全不干预’策略的净效益 (这个永远是 0)
    none_net_benefit = np.zeros_like(threshold_range)

    # 把结果整理到字典里
    dca_results = {
        'thresholds': threshold_range,
        'model_net_benefit': np.array(model_net_benefit), # 转成 numpy 数组
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

def plot_calibration_curves(model_names, y_true, y_pred_probas, save_path=None, n_bins=10):
    """
    画出多个模型的校准曲线 (可靠性曲线)，使用 tab10 颜色。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # --- ADDED: Get the tab10 colormap ---
    colors = plt.cm.get_cmap('tab10', len(model_names))
    # --- ---------------------------- ---

    ax.plot([0, 1], [0, 1], "k:", label="完美校准线")

    for i, model_name in enumerate(model_names):
        y_prob = y_pred_probas[i] if i < len(y_pred_probas) else None

        if y_prob is None:
             logger.warning(f"模型 {model_name} 缺少有效概率数据，跳过校准曲线绘制。")
             # Optionally plot a placeholder or just skip
             # ax.plot([0], [0], label=f"{model_name} (Data N/A)", color=colors(i), linestyle=':')
             continue

        # --- MODIFIED: Get the display object and set the line color ---
        display = CalibrationDisplay.from_predictions(
             y_true,
             y_prob,
             n_bins=n_bins,
             name=model_name,
             ax=ax
             # strategy='uniform' # or 'quantile' - uncomment if needed
        )
        # Set the color of the line plotted by CalibrationDisplay
        display.line_.set_color(colors(i))
        # Optional: Set the color of the bars in the histogram if needed
        # display.ax_hist.get_children()[i].set_color(colors(i)) # This might be fragile
        # --- -------------------------------------------------------- ---

    ax.set_xlabel("模型预测的平均概率")
    ax.set_ylabel("实际正例的比例")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title('校准曲线 (可靠性图)')
    ax.grid(True, alpha=0.3)

    # --- Saving logic (no changes needed here) ---
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"校准曲线图已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存校准曲线失败: {e}")
    else:
        plt.show()

    plt.close() # Close the figure

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
    plt.legend(loc="lower left", frameon=True, fancybox=True, 
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


def test_models(models_info, X_test, y_test, results_dir=None, n_bootstraps_auc=1000):
    """
    测试多个已加载的模型，比较性能，计算 AUC 的 95% CI，并保存 DeLong 检验结果。

    参数:
        models_info (dict): 字典，键是模型名，值是包含 'model', 'scaler', 'threshold' 的子字典。
        X_test (pd.DataFrame or np.ndarray): 测试集特征。
        y_test (pd.Series or np.ndarray): 测试集真实标签。
        results_dir (str, 可选): 保存结果文件的目录。默认不保存。
        n_bootstraps_auc (int, 可选): 计算 AUC CI 用的 Bootstrap 次数。默认 1000。

    返回:
        dict: 包含所有测试结果的字典，例如:
              {'metrics_summary': DataFrame,      # 单个模型指标总结
               'delong_comparison_summary': DataFrame, # DeLong 比较结果
               'auc_results': dict,             # 各模型 AUC 点估计和 CI
               'y_pred_probas': list,           # 各模型预测概率列表 (可能含 None)
               'thresholds_used': list,         # 各模型使用的阈值列表
               'dca_results': list}             # 各模型 DCA 计算结果列表
    """
    model_names = list(models_info.keys())
    logger.info(f"开始评估以下模型: {model_names}")

    # 如果指定了结果目录，确保它存在
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    # 初始化列表和字典来存储结果
    y_pred_probas_positive = [] # 存每个模型预测的正类概率 (一维)
    y_pred_labels = []          # 存每个模型基于阈值的预测标签 (0/1)
    metrics_list = []           # 存每个模型的详细性能指标字典
    thresholds_used = []        # 存每个模型实际用的阈值
    dca_results_list = []       # 存每个模型的 DCA 结果字典
    model_errors = {}           # 记录每个模型在评估中是否出错
    auc_results = {}            # 存每个模型的 AUC 点估计和 CI
    # 确保 y_test 是 numpy 数组，方便后续计算
    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)

    # 1. 挨个评估每个模型
    for model_name in model_names:
        logger.info(f"正在评估模型: {model_name}")
        # 从 models_info 获取模型、scaler 和阈值
        model = models_info[model_name]['model']
        scaler = models_info[model_name]['scaler']
        threshold = models_info[model_name]['threshold'] # 用加载时得到的阈值
        thresholds_used.append(threshold) # 记录一下

        # 标准化测试集特征
        X_test_scaled = scaler.transform(X_test)

        # 获取预测概率 (确保获取的是正类的概率，通常是第二列)
        y_pred_proba_all = model.predict_proba(X_test_scaled)
        if y_pred_proba_all.ndim == 2 and y_pred_proba_all.shape[1] >= 2:
            y_pred_proba = y_pred_proba_all[:, 1] # 假设正类是索引 1
        elif y_pred_proba_all.ndim == 1: # 处理可能直接返回1D数组的模型
            y_pred_proba = y_pred_proba_all
        else:
            logger.error(f"模型 {model_name} 的 predict_proba 输出格式不符合预期: {y_pred_proba_all.shape}")
            # Handle error appropriately, maybe skip saving or raise error
            y_pred_proba = None # Mark as None if format is wrong

        # --- 保存预测概率和真实标签 ---
        if results_dir and y_pred_proba is not None:
            try:
                pkl_save_path = os.path.join(results_dir, f"{model_name}_test_results.pkl")
                results_to_save = {'ensemble_probs': y_pred_proba, 'true_labels': y_test_np}
                with open(pkl_save_path, 'wb') as f:
                    pickle.dump(results_to_save, f)
                logger.info(f"  模型 {model_name} 的预测概率和标签已保存至: {pkl_save_path}")
            except Exception as e:
                logger.error(f"  保存模型 {model_name} 的 .pkl 文件失败: {e}")
        # --- ----------------------- ---

        # Append the 1D probability array (or None if failed)
        y_pred_probas_positive.append(y_pred_proba)

        # --- 计算 AUC 和它的 95% CI (仅当 y_pred_proba 有效时) ---
        if y_pred_proba is not None:
            auc_point, auc_lower, auc_upper = calculate_auc_ci(
                y_test_np, # 用 numpy 格式的真实标签
                y_pred_proba, # 用处理好的一维概率
                n_bootstraps=n_bootstraps_auc,
                seed=42 # 固定种子
            )
            # 把 AUC 结果存到字典里
            auc_results[model_name] = {'point': auc_point, 'lower': auc_lower, 'upper': auc_upper}
            auc_ci_str = f"{auc_point:.3f} ({auc_lower:.3f}-{auc_upper:.3f})" if not np.isnan(auc_point) else "N/A"

            # 根据阈值计算预测标签 (0 或 1)
            y_pred = (y_pred_proba >= threshold).astype(int)
            y_pred_labels.append(y_pred)

            # 计算各种性能指标
            metrics = calculate_metrics(y_test_np, y_pred_proba, threshold=threshold)
            metrics_list.append(metrics)

            # 计算决策曲线分析结果
            dca_result = decision_curve_analysis(y_test_np, y_pred_proba)
            dca_results_list.append(dca_result)

            # 打印当前模型的简要结果 (包含 AUC CI)
            logger.info(f"  模型: {model_name}, AUC (95% CI): {auc_ci_str}, 使用阈值: {threshold:.4f}")
            logger.info(f"  指标: 准确率={metrics['Accuracy']:.4f}, 敏感性={metrics['Sensitivity']:.4f}, 特异性={metrics['Specificity']:.4f}")

            model_errors[model_name] = None # 表示这个模型成功处理了
        else:
            # Handle the case where y_pred_proba is None (e.g., due to format error)
            auc_results[model_name] = {'point': np.nan, 'lower': np.nan, 'upper': np.nan}
            y_pred_labels.append(None)
            metrics_list.append({k: np.nan for k in ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'Threshold', 'TP', 'FN', 'FP', 'TN']})
            dca_results_list.append({'thresholds': np.arange(0.01, 1.0, 0.01),
                                      'model_net_benefit': np.full(99, np.nan),
                                      'all_net_benefit': np.full(99, np.nan),
                                      'none_net_benefit': np.zeros(99)})
            model_errors[model_name] = "Predict_proba output format error"
            logger.error(f"  模型 {model_name} 因概率格式错误跳过指标计算。")


    # --- 模型评估循环结束 ---

    # 筛选出那些成功跑完评估的模型及其结果
    valid_indices = [i for i, prob in enumerate(y_pred_probas_positive) if prob is not None]
    valid_model_names = [model_names[i] for i in valid_indices]
    valid_y_pred_probas = [y_pred_probas_positive[i] for i in valid_indices]
    valid_y_pred_labels = [y_pred_labels[i] for i in valid_indices]


    # 2. 模型性能比较 (DeLong 测试，只比较有效的模型)
    logger.info("开始进行模型间性能比较 (DeLong Test)...")
    delong_results_list = [] # 存 DeLong 测试结果的列表
    # 两两比较所有有效的模型
    for i in range(len(valid_model_names)):
        for j in range(i + 1, len(valid_model_names)):
            model1_name = valid_model_names[i]
            model2_name = valid_model_names[j]
            logger.info(f"  比较: {model1_name} vs {model2_name}")
            # 调用 DeLongTest 类进行计算
            delong_test = DelongTest(valid_y_pred_probas[i], valid_y_pred_probas[j], y_test_np)
            result = delong_test.result # 获取计算结果字典

            # 把结果整理成适合存入 DataFrame 的格式
            delong_result_row = {
                '模型 1': model1_name,
                '模型 2': model2_name,
                'AUC 1': result.get('AUC_1', np.nan), # 安全获取
                'AUC 2': result.get('AUC_2', np.nan),
                'AUC 差值 (1-2)': result.get('delta_AUC', np.nan),
                'Z 分数': result.get('z_score', np.nan),
                'P 值': result.get('p_value', np.nan),
                '是否显著 (p<0.05)': result.get('significant', False),
                '错误信息': result.get('error', None) # 如果有错误信息也记录下来
            }
            delong_results_list.append(delong_result_row)

            # 在日志里打印比较结论
            if 'error' in result and result['error']:
                    logger.warning(f"    DeLong 测试跳过: {result['error']}")
            elif np.isnan(result.get('p_value', np.nan)):
                    logger.warning(f"    DeLong 测试 P 值无法计算。")
            elif result.get('significant', False):
                better_model = model1_name if result.get('delta_AUC', 0) > 0 else model2_name
                logger.info(f"    结果显著: {better_model} 的 AUC 显著更高 (p={result['p_value']:.4f})")
            else:
                logger.info(f"    结果不显著: 两个模型的 AUC 无显著差异 (p={result['p_value']:.4f})")

    # 4. 绘图
    # 定义图片保存路径
    roc_save_path = os.path.join(results_dir, "roc_curves.png")
    calibration_save_path = os.path.join(results_dir, "calibration_curves.png")
    dca_save_path = os.path.join(results_dir, "dca_curves.png")

    # 画 ROC 曲线 (传入 auc_results 包含 CI 信息)
    # plot_roc_curves(model_names, y_test_np, y_pred_probas_positive, auc_results, save_path=roc_save_path)
    plot_roc_curves(model_names, y_test_np, y_pred_probas_positive, save_path=roc_save_path)
    # 画校准曲线
    plot_calibration_curves(model_names, y_test_np, y_pred_probas_positive, save_path=calibration_save_path)
    # 画决策曲线
    plot_dca_curves(model_names, dca_results_list, save_path=dca_save_path)

    # 5. 汇总结果并保存到 CSV 文件
    # (a) 单个模型的性能指标总结表
    metrics_summary_list = []
    for i, model_name in enumerate(model_names):
        # 获取 AUC 和 CI 结果
        auc_res = auc_results.get(model_name, {'point': np.nan, 'lower': np.nan, 'upper': np.nan})
        # 格式化 AUC (95% CI) 字符串
        auc_ci_str = f"{auc_res['point']:.3f} ({auc_res['lower']:.3f}-{auc_res['upper']:.3f})"


        # 从 metrics_list 安全地获取指标字典
        current_metrics = metrics_list[i] if i < len(metrics_list) else {}
        # 获取使用的阈值
        threshold_val = thresholds_used[i] if i < len(thresholds_used) else np.nan
        # 获取错误信息
        error_msg = model_errors.get(model_name, None)

        # 添加到汇总列表
        metrics_summary_list.append({
            '模型': model_name,
            'AUC (95% CI)': auc_ci_str, # 使用格式化后的字符串
            '准确率': current_metrics.get('Accuracy', np.nan), # 用 .get() 安全获取
            '敏感性': current_metrics.get('Sensitivity', np.nan),
            '特异性': current_metrics.get('Specificity', np.nan),
            'PPV': current_metrics.get('PPV', np.nan),
            'NPV': current_metrics.get('NPV', np.nan),
            'F1分数': current_metrics.get('F1', np.nan), # F1 也加进来
            '使用阈值': threshold_val,
            '评估错误': error_msg # 加入错误信息列
        })
    # 转换成 DataFrame
    metrics_summary = pd.DataFrame(metrics_summary_list)

    # (b) DeLong 测试的两两比较结果表
    delong_summary_df = pd.DataFrame(delong_results_list) # 直接用之前收集的列表创建


    metrics_summary_path = os.path.join(results_dir, "performance_metrics.csv")
    delong_summary_path = os.path.join(results_dir, "model_comparison_delong.csv")

    metrics_summary.to_csv(metrics_summary_path, index=False, float_format='%.4f')
    logger.info(f"性能指标汇总表已保存至: {metrics_summary_path}")

    delong_summary_df.to_csv(delong_summary_path, index=False, float_format='%.4f')
    logger.info(f"DeLong 检验比较结果已保存至: {delong_summary_path}")

    # 6. 打印总结到控制台
    print("\n--- 测试集模型性能指标汇总 ---")
    # 使用 to_string 打印完整表格
    print(metrics_summary.to_string(index=False, float_format='%.4f', na_rep='N/A'))

    print("\n--- 测试集模型两两比较 (DeLong Test) 结果 ---")
    print(delong_summary_df.to_string(index=False, float_format='%.4f', na_rep='N/A'))

    # 返回包含所有详细结果的字典
    return {
        'metrics_summary': metrics_summary,
        'delong_comparison_summary': delong_summary_df,
        'auc_results': auc_results, # 包含点估计和 CI
        'y_pred_probas': y_pred_probas_positive, # 预测概率列表 (可能含 None)
        'thresholds_used': thresholds_used, # 使用的阈值列表
        'dca_results': dca_results_list # DCA 结果列表
    }


def main():
    """主函数，执行整个测试流程"""
    # --- 配置区 ---
    config = {
        # 测试数据路径 (需要改成你自己的路径!)
        'test_features_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/radiomics/final_test_features.csv',
        'test_labels_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_labels.csv',
        # 保存结果的目录 (需要改成你自己的路径!)
        'results_dir': '/home/vipuser/Desktop/Classification/test_results_radiomics',
        # 训练好的模型、scaler、阈值所在的目录 (需要改成你自己的路径!)
        'models_dir': '/home/vipuser/Desktop/Classification/train_results_radiomics',
        # 要测试哪些模型 (名字要和训练时保存的文件名对应)
        'models_to_test': ['LR', 'SVM', 'MLP']
    }

    # 创建保存结果的目录 (如果不存在的话)
    os.makedirs(config['results_dir'], exist_ok=True)

    # 加载测试数据
    # 注意：这里去掉了 try-except，加载失败会报错
    X_test, y_test = load_data(config['test_features_path'], config['test_labels_path'])

    # 加载需要测试的模型、scaler 和阈值
    models_info = {} # 用字典存储加载好的信息
    logger.info("开始加载模型文件...")
    for model_type in config['models_to_test']:
        model_dir = os.path.join(config['models_dir'], model_type) # 模型自己的子目录
        # 构造模型、scaler、阈值文件的完整路径 (文件名用小写)
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_type.lower()}_scaler.pkl")
        threshold_path = os.path.join(model_dir, f"{model_type.lower()}_threshold.pkl")

        # 检查必需的模型和 scaler 文件是否存在
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # 注意：这里去掉了 try-except，加载失败会报错
            model, scaler, threshold = load_model(model_path, scaler_path, threshold_path)
            # 把加载好的东西存起来
            models_info[model_type] = {
                'model': model,
                'scaler': scaler,
                'threshold': threshold
            }
            logger.info(f"  成功加载模型: {model_type}")
        else:
            # 如果文件不全，打印警告并跳过这个模型
            logger.warning(f"模型 {model_type} 的文件不完整或不存在，将跳过评估。请检查路径:")
            logger.warning(f"  模型路径: {model_path} (是否存在: {os.path.exists(model_path)})")
            logger.warning(f"  Scaler路径: {scaler_path} (是否存在: {os.path.exists(scaler_path)})")
            # 也可以检查阈值文件是否存在，但不作为跳过的条件
            # logger.warning(f"  阈值路径: {threshold_path} (是否存在: {os.path.exists(threshold_path)})")

    logger.info(f"成功加载 {len(models_info)} 个模型，开始进行测试评估...")
    test_models(
        models_info,
        X_test,
        y_test,
        results_dir=config['results_dir']
    )
    logger.info("所有模型测试评估完成。")

if __name__ == "__main__":
    main()
    logger.info("测试脚本运行结束。")