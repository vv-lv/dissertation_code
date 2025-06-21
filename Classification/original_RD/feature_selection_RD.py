import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, kruskal
from skrebate import ReliefF
import random
from datetime import datetime
import re
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import RFE, RFECV # 导入 RFECV
from sklearn.model_selection import StratifiedKFold # 推荐用于分类问题的CV

def set_all_seeds(seed=42):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    # Optional: os.environ['PYTHONHASHSEED'] = str(seed)

def create_visualization_dir(base_dir):
    """Create a directory for saving visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(base_dir, f"feature_selection_vis")
    # vis_dir = os.path.join(base_dir, f"feature_selection_vis_{timestamp}")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def plot_feature_correlation_heatmap(X, selected_features, vis_dir):
    """Plot and save correlation heatmap for final selected features only"""
    # Filter dataframe to include only the selected features
    X_selected = X[selected_features]
    
    # Calculate correlation matrix
    correlation_matrix = X_selected.corr(method='spearman')
    
    plt.figure(figsize=(20, 16))
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    plt.title("Correlation Heatmap of Final Selected Features")
        
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "final_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved final correlation heatmap")

def categorize_features(features):
    """
    Categorize features based on common patterns in radiomics feature names
    
    Returns:
    dict: Dictionary with categories as keys and feature counts as values
    """
    categories = {
        'Shape': 0,
        'First Order': 0,
        'GLCM': 0,
        'GLRLM': 0,
        'GLSZM': 0,
        'NGTDM': 0,
        'GLDM': 0,
        'Wavelet': 0,
        'LoG': 0,
        'Other': 0
    }
    
    pattern_mapping = {
        r'shape': 'Shape',
        r'firstorder': 'First Order',
        r'glcm': 'GLCM',
        r'glrlm': 'GLRLM',
        r'glszm': 'GLSZM',
        r'ngtdm': 'NGTDM',
        r'gldm': 'GLDM',
        r'wavelet': 'Wavelet',
        r'log': 'LoG'
    }
    
    for feature in features:
        feature_lower = feature.lower()
        categorized = False
        
        for pattern, category in pattern_mapping.items():
            if re.search(pattern, feature_lower):
                categories[category] += 1
                categorized = True
                break
                
        if not categorized:
            categories['Other'] += 1
    
    # Remove categories with zero count
    return {k: v for k, v in categories.items() if v > 0}

def plot_feature_types(selected_features, vis_dir):
    """Plot summary of feature types in the final selected features"""
    # Categorize features
    categories = categorize_features(selected_features)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(categories.keys(), categories.values(), color='skyblue')
    
    # Add count labels on top of each bar
    for i, (category, count) in enumerate(categories.items()):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    plt.xlabel('Feature Category')
    plt.ylabel('Number of Features')
    plt.title('Feature Types in Final Selection')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(vis_dir, "feature_types_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved feature types summary plot")

def find_best_alpha_with_cv(X, y, alphas=None, is_classification=False, n_folds=5, random_state=42):
    """
    执行交叉验证寻找最优alpha值
    
    Parameters:
    -----------
    X : pandas.DataFrame
        特征矩阵
    y : array-like
        目标变量
    alphas : array-like, default=None
        要评估的alpha值列表，若为None则自动生成
    is_classification : bool, default=False
        是否为分类任务
    n_folds : int, default=5
        交叉验证折数
    random_state : int, default=42
        随机种子
        
    Returns:
    --------
    best_alpha : float
        最优alpha值
    alphas : array-like
        评估的alpha值列表
    mean_metrics : array-like
        每个alpha的平均性能指标
    std_metrics : array-like
        每个alpha的性能指标标准差
    """
    # 如果没有提供alpha值列表，则创建默认列表
    if alphas is None:
        alphas = np.logspace(-5, 1, 50)
    
    n_alphas = len(alphas)
    
    # 设置交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # 存储每个alpha在每个折上的度量值
    all_metrics = np.zeros((n_folds, n_alphas))
    
    # 对每个折进行模型训练和评估
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for alpha_idx, alpha in enumerate(alphas):
            try:
                if is_classification:
                    # 分类问题
                    model = LogisticRegression(penalty='l1', solver='liblinear', 
                                              C=1/alpha, max_iter=10000, random_state=random_state)
                    model.fit(X_train, y_train)
                    
                    if len(np.unique(y)) == 2:  # 二分类
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        metric = roc_auc_score(y_val, y_pred_proba)
                    else:  # 多分类
                        y_pred = model.predict(X_val)
                        metric = np.mean(y_pred == y_val)  # 准确率
                else:
                    # 回归问题
                    model = Lasso(alpha=alpha, max_iter=10000, random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    # 使用负均方误差（值越高越好，与分类指标方向一致）
                    metric = -np.mean((y_val - y_pred) ** 2)
                
                all_metrics[fold_idx, alpha_idx] = metric
            except Exception as e:
                print(f"Alpha {alpha} evaluation error: {str(e)}")
                # 对于分类任务，设置一个较低的度量值表示较差的性能
                all_metrics[fold_idx, alpha_idx] = 0.5 if is_classification and len(np.unique(y)) == 2 else 0
    
    # 计算每个alpha的平均度量值和标准差
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)
    
    # 找到最优alpha（分类问题是越高越好，回归问题也是越高越好因为使用的是负MSE）
    best_alpha_idx = np.argmax(mean_metrics)
    best_alpha = alphas[best_alpha_idx]
    
    return best_alpha, alphas, mean_metrics, std_metrics

def plot_lasso_cv_error(alphas, mean_metrics, std_metrics, best_alpha, vis_dir, is_classification=False):
    """
    绘制带误差条的交叉验证误差图
    
    Parameters:
    -----------
    alphas : array-like
        评估的alpha值列表
    mean_metrics : array-like
        每个alpha的平均性能指标
    std_metrics : array-like
        每个alpha的性能指标标准差
    best_alpha : float
        最优alpha值
    vis_dir : str
        保存可视化结果的目录
    is_classification : bool, default=False
        是否为分类任务
    """
    plt.figure(figsize=(12, 6))
    
    # 确定度量指标名称
    if is_classification:
        if len(np.unique(mean_metrics)) > 2:  # 近似判断是否为二分类
            metric_name = "AUC"
        else:
            metric_name = "Accuracy"
        plt.title(f'Cross-Validation {metric_name} for LogisticRegression with L1 Penalty')
        plt.ylabel(f'{metric_name} Score')
    else:
        plt.title('Cross-Validation Error for LASSO Regression')
        plt.ylabel('MSE')
    
    # 主度量曲线（红色实线）
    plt.semilogx(alphas, mean_metrics, 'r-', linewidth=2)
    
    # 添加误差条（浅蓝色垂直线）
    for i, alpha in enumerate(alphas):
        plt.vlines(alpha, mean_metrics[i] - std_metrics[i], 
                  mean_metrics[i] + std_metrics[i], 
                  color='lightblue', alpha=0.7, linewidth=3)
    
    # 标记最优alpha
    plt.axvline(x=best_alpha, color='black', linestyle='--')
    
    # 在图底部标注最优lambda值
    if is_classification:
        annotation_text = f"λ = {best_alpha:.6e} (最高{metric_name})"
    else:
        annotation_text = f"λ = {best_alpha:.6e} (最低MSE)"
        
    plt.annotate(annotation_text, 
                xy=(0.7, 0.02), 
                xycoords='figure fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    
    # 其他图形设置
    plt.xlabel('Lambda (log scale)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(os.path.join(vis_dir, "lasso_cv_error.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存了LASSO交叉验证误差图，最优lambda = {best_alpha:.6f}")

def plot_lasso_path(X, y, best_alpha, vis_dir, is_classification=False, random_state=42):
    """
    绘制LASSO系数路径图，聚焦于最优lambda附近
    
    Parameters:
    -----------
    X : pandas.DataFrame
        特征矩阵
    y : array-like
        目标变量
    best_alpha : float
        通过交叉验证得到的最优alpha值
    vis_dir : str
        保存可视化结果的目录
    is_classification : bool, default=False
        是否为分类任务
    random_state : int, default=42
        随机种子
    """
    # 创建适当的alpha值范围用于路径图（比交叉验证用的更密集）
    # 以best_alpha为中心，扩展范围
    log_best_alpha = np.log10(best_alpha)
    alpha_min = 10 ** max(-5, log_best_alpha - 2.0)
    alpha_max = 10 ** min(2, log_best_alpha + 2.0)
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 100)
    
    # 计算系数路径
    if is_classification:
        coefs = []
        for a in alphas:
            model = LogisticRegression(penalty='l1', solver='liblinear', 
                                      C=1/a, max_iter=10000, random_state=random_state)
            model.fit(X, y)
            coefs.append(model.coef_.ravel())
    else:
        # 回归问题
        coefs = []
        for a in alphas:
            model = Lasso(alpha=a, max_iter=10000, random_state=random_state)
            model.fit(X, y)
            coefs.append(model.coef_)
    
    coefs = np.array(coefs)
    
    # 找到best_alpha对应的索引
    best_idx = np.argmin(np.abs(alphas - best_alpha))
    
    # 显示最重要的特征（在最优lambda下非零的系数）
    log_alphas = np.log10(alphas)
    best_coefs = coefs[best_idx]
    nonzero_indices = np.where(np.abs(best_coefs) > 1e-5)[0]
    print(f"非零系数数量: {len(nonzero_indices)}")
    
    if len(nonzero_indices) > 0:
        plt.figure(figsize=(14, 10))
        plt.title(f"Lasso Path - Top {len(nonzero_indices)} Significant Coefficients")
        
        # 只绘制在最优lambda处非零的系数
        for idx in nonzero_indices:
            plt.plot(log_alphas, coefs[:, idx], linewidth=2, 
                    label=f"Feature {idx}" if len(nonzero_indices) < 10 else None)
        
        plt.axvline(x=log_best_alpha, color='k', linestyle='--', 
                   label=f'Best log(lambda) = {log_best_alpha:.2f}')
        
        # 聚焦在最优lambda前后约1.5个数量级的范围
        x_min = max(log_alphas[0], log_best_alpha - 2)
        x_max = min(log_alphas[-1], log_best_alpha + 2)
        plt.xlim(x_min, x_max)
        
        # 动态调整y轴范围，聚焦于重要区域
        x_range_indices = (log_alphas >= x_min) & (log_alphas <= x_max)
        visible_coefs = coefs[x_range_indices][:, nonzero_indices]
        
        if visible_coefs.size > 0:  # 确保有可见的系数
            y_min = np.min(visible_coefs) * 1.1  # 留出10%的边距
            y_max = np.max(visible_coefs) * 1.1
            plt.ylim(y_min, y_max)
        
        annotation_text = f"λ = {best_alpha:.6e} (保留{len(nonzero_indices)}个非零系数)"
        plt.annotate(annotation_text, 
                    xy=(0.7, 0.02), 
                    xycoords='figure fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        
        plt.xlabel('Log(lambda)')
        plt.ylabel('Coefficients')
        if len(nonzero_indices) < 10:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "lasso_paths.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 保存了LASSO路径图，使用最优lambda = {best_alpha:.6f}")

def plot_feature_selection_summary(original_count, var_count, corr_count, 
                                   kw_count, relief_count, lasso_count, rfe_count, vis_dir):
    """Plot summary of feature counts after each selection step"""
    steps = ['Original', 'Variance\nFiltering', 'Correlation\nFiltering', 
             'KW Test', 'Relief', 'LASSO', 'RFE']
    counts = [original_count, var_count, corr_count, kw_count, relief_count, lasso_count, rfe_count]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, counts, marker='o', linestyle='-', linewidth=2, markersize=10)
    
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.02, str(count), ha='center')
    
    plt.xlabel('Feature Selection Step')
    plt.ylabel('Number of Features')
    plt.title('Feature Count Throughout Selection Pipeline')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_selection_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved feature selection summary plot")
def feature_selection_pipeline(X, y,
                               variance_threshold=0.01,
                               correlation_threshold=0.8,
                               kw_p_value_threshold=0.05,      # <--- 修改：使用 p 值阈值代替 k
                               relief_threshold_method='above_mean', # <--- 修改：使用阈值方法代替 k ('above_mean', 'keep_positive', 'none')
                               # lasso_k 和 rfe_k 参数被移除，因为它们是自动的
                               random_state=42,
                               visualization_dir=None):
    """
    Complete feature selection pipeline with visualizations and automatic feature count determination
    for KW, Relief, LASSO, and RFE(CV).

    Parameters:
    X - Feature matrix (DataFrame)
    y - Target variable (Series or ndarray)
    variance_threshold - Variance threshold for feature removal
    correlation_threshold - Correlation threshold for redundancy removal
    kw_p_value_threshold - P-value threshold for Kruskal-Wallis test (e.g., 0.05)
    relief_threshold_method - Method to determine ReliefF threshold ('above_mean', 'keep_positive', 'none')
    random_state - Random seed for reproducibility
    visualization_dir - Directory to save visualizations

    Returns:
    selected_features - Final list of selected features
    feature_selection_results - Results from each step
    visualization_dir - Path to the visualization directory
    """
    # Set seeds for reproducibility
    set_all_seeds(random_state)

    # Create visualization directory if needed
    if visualization_dir is None:
        visualization_dir = create_visualization_dir(os.getcwd()) # 确保调用了创建目录的函数

    feature_selection_results = {}
    print(f"原始特征数量: {X.shape[1]}")
    original_count = X.shape[1]
    X_processed = X.copy()

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)
    feature_selection_results['scaler'] = scaler # 保存 scaler

    # Step 1: Remove low variance features
    # ... (方差过滤部分不变) ...
    var_selector = VarianceThreshold(threshold=variance_threshold)
    var_selector.fit(X_scaled)
    mask = var_selector.get_support()
    X_var_selected = X_scaled.loc[:, mask]
    removed_by_variance = X_scaled.columns[~mask].tolist()
    feature_selection_results['removed_by_variance'] = removed_by_variance
    print(f"低方差特征移除后剩余: {X_var_selected.shape[1]}")
    var_count = X_var_selected.shape[1]
    if var_count == 0:
        print("警告: 方差过滤后没有特征剩余！流程终止。")
        plot_feature_selection_summary(original_count, var_count, 0, 0, 0, 0, 0, visualization_dir)
        return [], feature_selection_results, visualization_dir


    # Step 2: Remove redundant features based on correlation
    # ... (相关性过滤部分不变) ...
    correlation_matrix = pd.DataFrame(np.zeros((X_var_selected.shape[1], X_var_selected.shape[1])),
                                     index=X_var_selected.columns, columns=X_var_selected.columns)
    for i, feature_i in enumerate(X_var_selected.columns):
        for j, feature_j in enumerate(X_var_selected.columns):
            if i >= j:
                corr_value, _ = spearmanr(X_var_selected[feature_i], X_var_selected[feature_j])
                correlation_matrix.loc[feature_i, feature_j] = corr_value
                correlation_matrix.loc[feature_j, feature_i] = corr_value

    selected_corr = []
    removed_by_correlation = []
    remaining_features = X_var_selected.columns.tolist()
    while len(remaining_features) > 0:
        selected_feature = remaining_features[0]
        selected_corr.append(selected_feature)
        highly_correlated_with_selected = []
        for feature in remaining_features[1:]:
            if abs(correlation_matrix.loc[selected_feature, feature]) > correlation_threshold:
                 removed_by_correlation.append((feature, selected_feature, correlation_matrix.loc[selected_feature, feature]))
                 highly_correlated_with_selected.append(feature)

        remaining_features = [f for f in remaining_features if f not in highly_correlated_with_selected]
        remaining_features.remove(selected_feature) # 移除当前选中的

    feature_selection_results['removed_by_correlation'] = removed_by_correlation
    X_corr_selected = X_var_selected[selected_corr]
    print(f"相关性筛选后剩余: {X_corr_selected.shape[1]}")
    corr_count = X_corr_selected.shape[1]
    if corr_count == 0:
        print("警告: 相关性过滤后没有特征剩余！流程终止。")
        plot_feature_selection_summary(original_count, var_count, corr_count, 0, 0, 0, 0, visualization_dir)
        return [], feature_selection_results, visualization_dir

    # Step 3a: Kruskal-Wallis test for classification problems (using p-value threshold)
    is_classification = len(np.unique(y)) < 10 or isinstance(y[0], (np.int64, np.int32, int, str)) # 更稳健的分类判断

    if is_classification:
        print(f"\n执行 KW 检验 (p < {kw_p_value_threshold})...")
        kw_scores = []
        for feature in X_corr_selected.columns:
            try:
                groups = [X_corr_selected[feature][y == label].values for label in np.unique(y)]
                # 检查每个组是否至少有一个样本
                if all(len(g) > 0 for g in groups):
                    statistic, p_value = kruskal(*groups)
                    kw_scores.append({'feature': feature, 'statistic': statistic, 'p_value': p_value})
                else:
                    # 如果某个类别没有样本，则无法进行KW检验，赋予一个无效的p值
                     kw_scores.append({'feature': feature, 'statistic': np.nan, 'p_value': 1.0})
            except ValueError as e:
                 print(f"  处理特征 '{feature}' 时 KW 检验出错: {e}. 跳过该特征。")
                 kw_scores.append({'feature': feature, 'statistic': np.nan, 'p_value': 1.0})


        # Filter based on p-value threshold
        kw_selected_info = [item for item in kw_scores if item['p_value'] < kw_p_value_threshold]
        kw_selected = [item['feature'] for item in kw_selected_info]

        # 按统计量排序（可选，主要用于结果记录）
        kw_selected_info.sort(key=lambda x: x['statistic'], reverse=True)
        feature_selection_results['kw_results'] = kw_selected_info

        if not kw_selected:
             print(f"警告: KW 检验 (p < {kw_p_value_threshold}) 未选择任何特征。可能阈值太严格或特征区分度不足。")
             # 决定是停止还是继续使用相关性过滤后的特征？这里选择继续，但发出警告。
             X_kw_selected = X_corr_selected
             kw_selected = X_corr_selected.columns.tolist()
             kw_count = X_kw_selected.shape[1]
             print(f"  将使用相关性过滤后的全部 {kw_count} 个特征进行下一步。")

        else:
            X_kw_selected = X_corr_selected[kw_selected]
            print(f"KW 检验 (p < {kw_p_value_threshold}) 筛选后剩余: {X_kw_selected.shape[1]}")
            kw_count = X_kw_selected.shape[1]

    else: # 回归任务
        kw_selected = X_corr_selected.columns.tolist()
        X_kw_selected = X_corr_selected
        feature_selection_results['kw_results'] = "回归任务中跳过KW检验"
        kw_count = X_kw_selected.shape[1]
        print(f"回归任务，跳过 KW 检验，保留 {kw_count} 个特征。")

    if kw_count == 0: # 再次检查，以防分类任务中未选出特征且未处理
        print("警告: KW 步骤后没有特征剩余！流程终止。")
        plot_feature_selection_summary(original_count, var_count, corr_count, kw_count, 0, 0, 0, visualization_dir)
        return [], feature_selection_results, visualization_dir


    # Step 3b: Relief feature selection (using threshold method)
    print(f"\n执行 ReliefF (阈值方法: {relief_threshold_method})...")
    # Fit ReliefF on all current features to get scores
    # n_features_to_select 设置为当前特征数，因为我们只关心分数
    relief = ReliefF(n_features_to_select=X_kw_selected.shape[1], n_neighbors=10, n_jobs=-1)
    relief.fit(X_kw_selected.values, y)
    feature_scores = relief.feature_importances_
    feature_score_map = dict(zip(X_kw_selected.columns, feature_scores))

    # Apply threshold method
    if relief_threshold_method == 'above_mean':
        mean_score = np.mean(feature_scores)
        relief_selected = [f for f, score in feature_score_map.items() if score > mean_score]
        print(f"  选择分数高于平均值 ({mean_score:.4f}) 的特征。")
    elif relief_threshold_method == 'keep_positive':
        relief_selected = [f for f, score in feature_score_map.items() if score > 0]
        print("  选择分数大于 0 的特征。")
    elif relief_threshold_method == 'none':
        relief_selected = X_kw_selected.columns.tolist()
        print("  未应用 ReliefF 阈值过滤。")
    else:
        print(f"  警告: 未知的 ReliefF 阈值方法 '{relief_threshold_method}'. 未执行过滤。")
        relief_selected = X_kw_selected.columns.tolist()

    if not relief_selected:
        print(f"警告: ReliefF (方法: {relief_threshold_method}) 未选择任何特征。可能阈值不适用。")
        # 同样，决定是停止还是继续？选择继续使用 KW 后的特征。
        X_relief_selected = X_kw_selected
        relief_selected = X_kw_selected.columns.tolist()
        relief_count = X_relief_selected.shape[1]
        print(f"  将使用 KW 步骤后的全部 {relief_count} 个特征进行下一步。")
    else:
        X_relief_selected = X_kw_selected[relief_selected]
        relief_count = X_relief_selected.shape[1]
        print(f"ReliefF (方法: {relief_threshold_method}) 筛选后剩余: {relief_count}")

    # 按分数排序用于记录
    sorted_relief_features = sorted(feature_score_map.items(), key=lambda item: item[1], reverse=True)
    feature_selection_results['relief_results'] = {
        'selected': relief_selected,
        'scores': dict(sorted_relief_features) # 保存排序后的分数
    }

    if relief_count == 0:
        print("警告: Relief 步骤后没有特征剩余！流程终止。")
        plot_feature_selection_summary(original_count, var_count, corr_count, kw_count, relief_count, 0, 0, visualization_dir)
        return [], feature_selection_results, visualization_dir

    # Step 4a: LASSO feature selection (Automatic via CV)
    print("\n执行 LASSO (通过 CV 自动选择)...")
    alphas_lasso = np.logspace(-5, 1, 50) # 保持CV的alpha范围
    best_alpha, checked_alphas, mean_metrics, std_metrics = find_best_alpha_with_cv(
        X_relief_selected, y, alphas_lasso, is_classification, random_state=random_state
    )
    print(f"  找到最优 Alpha (Lambda): {best_alpha:.6f}")

    plot_lasso_cv_error(
        checked_alphas, mean_metrics, std_metrics, best_alpha, visualization_dir, is_classification
    )
    plot_lasso_path(
        X_relief_selected, y, best_alpha, visualization_dir, is_classification, random_state
    )

    # Fit final LASSO model with best alpha
    if is_classification:
        # 增加 max_iter 并考虑 class_weight
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1/best_alpha,
                                   max_iter=10000, random_state=random_state,
                                   class_weight='balanced' if len(np.unique(y))==2 else None) # 二分类加 balanced
    else:
        lasso = Lasso(alpha=best_alpha, max_iter=10000, random_state=random_state)

    lasso.fit(X_relief_selected, y)

    # Get coefficients and select non-zero ones
    if hasattr(lasso, 'coef_'):
        # 处理多分类逻辑回归的 coef_ 形状 (n_classes, n_features)
        if lasso.coef_.ndim > 1:
             # 对每个特征，取所有类别系数绝对值的最大值或总和作为重要性
             # 这里使用最大值可能更合理，表示该特征对至少一个类别的区分度
             importance = np.max(np.abs(lasso.coef_), axis=0)
             # 或者求和： importance = np.sum(np.abs(lasso.coef_), axis=0)
        else:
             importance = np.abs(lasso.coef_.ravel()) # 确保是一维
    # else: # 应该总有 coef_ 对于 LR 和 Lasso
    #     importance = np.zeros(X_relief_selected.shape[1]) # 以防万一

    nonzero_threshold = 1e-5
    nonzero_indices = np.where(importance > nonzero_threshold)[0]
    lasso_selected = X_relief_selected.columns[nonzero_indices].tolist()

    if not lasso_selected:
        print(f"警告: LASSO (alpha={best_alpha:.4f}) 未选择任何特征。可能正则化过强或特征信号弱。")
        # 选择继续使用 Relief 后的特征
        X_lasso_selected = X_relief_selected
        lasso_selected = X_relief_selected.columns.tolist()
        lasso_count = X_lasso_selected.shape[1]
        print(f"  将使用 Relief 步骤后的全部 {lasso_count} 个特征进行下一步。")
    else:
        X_lasso_selected = X_relief_selected[lasso_selected]
        lasso_count = X_lasso_selected.shape[1]
        print(f"LASSO (alpha={best_alpha:.4f}) 筛选后剩余: {lasso_count}")


    lasso_coeffs = dict(zip(X_relief_selected.columns, importance))
    sorted_lasso_coeffs = sorted(lasso_coeffs.items(), key=lambda item: item[1], reverse=True)
    feature_selection_results['lasso_results'] = {
        'selected': lasso_selected,
        'best_alpha': best_alpha,
        'coefficients': dict(sorted_lasso_coeffs) # 保存排序后的系数绝对值
    }

    if lasso_count == 0:
        print("警告: LASSO 步骤后没有特征剩余！流程终止。")
        plot_feature_selection_summary(original_count, var_count, corr_count, kw_count, relief_count, lasso_count, 0, visualization_dir)
        return [], feature_selection_results, visualization_dir

    # Step 4b: Recursive Feature Elimination with Cross-Validation (RFECV - Automatic)
    print("\n执行 RFECV (自动确定最佳特征数)...")
    min_features_to_select = max(1, min(5, X_lasso_selected.shape[1] // 2)) # 动态设置最小值，至少为1，最多5个或一半

    if X_lasso_selected.shape[1] < min_features_to_select:
         print(f"警告: LASSO后剩余特征数({X_lasso_selected.shape[1]}) 不足 RFECV 最小要求({min_features_to_select})。将直接使用 LASSO 结果。")
         rfe_selected = lasso_selected
         rfe_optimal_n_features = len(rfe_selected)
         feature_selection_results['rfe_results'] = {
             'selected': rfe_selected,
             'ranking': {feat: 1 for feat in rfe_selected},
             'optimal_n_features': rfe_optimal_n_features,
             'cv_scores': None,
             'skipped': True
         }
    elif X_lasso_selected.shape[1] <= 1: # 如果只剩一个特征，RFECV也无意义
         print(f"警告: LASSO后只剩 {X_lasso_selected.shape[1]} 个特征。跳过 RFECV。")
         rfe_selected = lasso_selected
         rfe_optimal_n_features = len(rfe_selected)
         feature_selection_results['rfe_results'] = {
             'selected': rfe_selected,
             'ranking': {feat: 1 for feat in rfe_selected},
             'optimal_n_features': rfe_optimal_n_features,
             'cv_scores': None,
             'skipped': True
         }
    else:
        if is_classification:
            # 确保使用平衡权重处理不平衡数据
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced', n_jobs=-1)
            cv_strategy = StratifiedKFold(5, shuffle=True, random_state=random_state)
            # 对于 AUC，确保是二分类问题
            if len(np.unique(y)) == 2:
                scoring_metric = 'roc_auc'
            else: # 多分类用 F1 或 Accuracy
                scoring_metric = 'f1_weighted' # 或者 'accuracy'
        # else:
        #     estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        #     cv_strategy = KFold(5, shuffle=True, random_state=random_state)
        #     scoring_metric = 'neg_root_mean_squared_error'

        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_strategy,
            scoring=scoring_metric,
            min_features_to_select=min_features_to_select,
            n_jobs=-1 # RFECV 本身也用 n_jobs
        )

        rfecv.fit(X_lasso_selected, y)
        rfe_selected = X_lasso_selected.columns[rfecv.support_].tolist()
        rfe_optimal_n_features = rfecv.n_features_
        print(f"RFECV 找到的最佳特征数量: {rfe_optimal_n_features}")

        feature_selection_results['rfe_results'] = {
            'selected': rfe_selected,
            'ranking': dict(zip(X_lasso_selected.columns, rfecv.ranking_)),
            'optimal_n_features': rfe_optimal_n_features,
            'cv_scores': rfecv.cv_results_.get('mean_test_score', None), # 安全获取
             'skipped': False
        }

        # 可选：绘制 RFECV 性能曲线 (需要确保 rfecv.cv_results_ 可用)
        if 'mean_test_score' in rfecv.cv_results_:
             try:
                 plt.figure(figsize=(10, 6))
                 scores = rfecv.cv_results_['mean_test_score']
                 # x轴是从 min_features_to_select 到 X_lasso_selected.shape[1]
                 num_features_tested = range(min_features_to_select, X_lasso_selected.shape[1] + 1)
                 # 检查 scores 和 num_features_tested 的长度是否匹配
                 if len(scores) == len(num_features_tested):
                     plt.plot(num_features_tested, scores)
                     plt.xlabel("Number of features selected")
                     plt.ylabel(f"Cross validation score ({scoring_metric})")
                     plt.axvline(x=rfe_optimal_n_features, color='r', linestyle='--',
                                 label=f'Optimal = {rfe_optimal_n_features}')
                     plt.legend()
                     plt.title('RFECV Performance vs Number of Features')
                     plt.grid(True, alpha=0.3)
                     plt.tight_layout()
                     plt.savefig(os.path.join(visualization_dir, "rfecv_performance.png"), dpi=300, bbox_inches='tight')
                     plt.close()
                     print("✓ Saved RFECV performance plot")
                 else:
                      print(f"警告: RFECV scores ({len(scores)}) 与测试的特征数量范围 ({len(num_features_tested)}) 不匹配，无法绘制性能图。")

             except Exception as plot_err:
                 print(f"绘制 RFECV 性能图时出错: {plot_err}")

    # 获取最终的特征集
    X_rfe_selected = X_lasso_selected[rfe_selected]
    rfe_count = X_rfe_selected.shape[1]
    final_selected_features = X_rfe_selected.columns.tolist()

    if rfe_count == 0:
        print("警告: RFECV 步骤后没有特征剩余！流程终止。")
        # 绘制最终的总结图
        plot_feature_selection_summary(original_count, var_count, corr_count,
                                   kw_count, relief_count, lasso_count, rfe_count,
                                   visualization_dir)
        return [], feature_selection_results, visualization_dir


    # Plot final summaries using the final selected features
    print("\n绘制最终结果图表...")
    plot_feature_selection_summary(original_count, var_count, corr_count,
                                   kw_count, relief_count, lasso_count, rfe_count,
                                   visualization_dir)

    print(f"最终选择的特征 ({len(final_selected_features)}): {', '.join(final_selected_features[:10])}...") # 显示多一点

    # 使用原始 X (未标准化的) 和最终选择的特征列表来绘制相关性热图
    plot_feature_correlation_heatmap(X_processed[final_selected_features], final_selected_features, visualization_dir)
    plot_feature_types(final_selected_features, visualization_dir)

    # 返回最终特征列表、每步结果和可视化目录
    return final_selected_features, feature_selection_results, visualization_dir

def apply_selected_features(original_df, selected_features):
    """Apply selected features to original data (including ID column)"""
    if 'ID' in original_df.columns:
        return original_df[['ID'] + selected_features]
    else:
        return original_df[selected_features]

# Main program
if __name__ == "__main__":
    # Set global random seed for reproducibility
    set_all_seeds(42)
    
    # Data paths
    data_dir = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics"
    train_features_path = os.path.join(data_dir, "train_features_RD.csv")
    test_features_path = os.path.join(data_dir, "test_features_RD.csv")
    train_labels_path = os.path.join(data_dir, "train_labels.csv")
    test_labels_path = os.path.join(data_dir, "test_labels.csv")
    
    # Output paths
    output_dir = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/RD"
    selected_features_path = os.path.join(output_dir, "selected_features.txt")
    final_train_path = os.path.join(output_dir, "final_train_features.csv")
    final_test_path = os.path.join(output_dir, "final_test_features.csv")

    # Create visualization directory
    vis_dir = create_visualization_dir(output_dir)

    # Load training data
    print("加载训练集数据...")
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)

    # Extract features and target, ensuring consistent data types
    X_train = train_features.drop('ID', axis=1)
    # 尝试转换非数值列为数值，无法转换的填充NaN或移除（这里选择填充0，但需谨慎）
    for col in X_train.columns:
         if X_train[col].dtype == 'object':
             try:
                 X_train[col] = pd.to_numeric(X_train[col])
             except ValueError:
                 print(f"警告：列 '{col}' 包含非数值数据，将尝试移除。如果重要请预处理。")
                 # 或者填充：X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                 X_train = X_train.drop(col, axis=1)


    y_train = train_labels['label'].values

    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}") # 打印更清晰的标签分布


    # Feature selection with visualizations - 使用新的参数
    print("\n执行特征选择流程 (自动确定数量)...")
    selected_features, selection_results, visualization_dir = feature_selection_pipeline(
        X_train, y_train,
        variance_threshold=0.1,         # 方差阈值
        correlation_threshold=0.7,      # 相关性阈值
        kw_p_value_threshold=0.05,      # KW p 值阈值
        relief_threshold_method='above_mean', # ReliefF 阈值方法
        random_state=42,
        visualization_dir=vis_dir
    )

    # 检查是否有特征被选出
    if not selected_features:
        print("\n特征选择流程未能选出任何特征。请检查参数或数据。")
    else:
        # Save selected features list
        with open(selected_features_path, "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        print(f"\n选定的特征 ({len(selected_features)}) 已保存到: {selected_features_path}")
        print(f"可视化结果已保存到: {visualization_dir}")

        # Apply selected features to training set
        final_train_features = apply_selected_features(train_features, selected_features)
        final_train_features.to_csv(final_train_path, index=False)
        print(f"筛选后的训练集已保存到: {final_train_path}")

        # Apply same features to test set
        print("\n加载并处理测试集数据...")
        try:
            test_features = pd.read_csv(test_features_path)
            print(f"原始测试集: {test_features.shape[0]} 样本, {test_features.shape[1]} 特征")

            # 检查测试集是否包含所有选定的特征
            missing_in_test = [f for f in selected_features if f not in test_features.columns]
            if missing_in_test:
                print(f"警告: 测试集中缺少 {len(missing_in_test)} 个在训练集中选定的特征: {missing_in_test}")
                # 处理缺失特征：是移除这些特征，还是尝试填充？这里选择移除
                selected_features_in_test = [f for f in selected_features if f in test_features.columns]
                print(f"  将只使用测试集中存在的 {len(selected_features_in_test)} 个选定特征。")
                if not selected_features_in_test:
                     print("错误：测试集中不存在任何选定的特征，无法生成最终测试集。")
                     final_test_features = pd.DataFrame({'ID': test_features['ID']}) # 只保留ID
                else:
                     final_test_features = apply_selected_features(test_features, selected_features_in_test)

            else:
                final_test_features = apply_selected_features(test_features, selected_features)

            final_test_features.to_csv(final_test_path, index=False)
            print(f"筛选后的测试集 ({final_test_features.shape[1]-1 if 'ID' in final_test_features else final_test_features.shape[1]} 特征) 已保存到: {final_test_path}")

        except FileNotFoundError:
            print(f"警告: 测试集特征文件未找到: {test_features_path}")
        except Exception as e:
            print(f"处理测试集时发生错误: {str(e)}")

    print("\n所有处理完成！")
    