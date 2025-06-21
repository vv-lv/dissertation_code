import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import torch
import pickle
import logging
from datetime import datetime

from models import create_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train')

def load_data(features_path, labels_path):
    """
    加载并准备好数据。

    参数:
        features_path (str): 特征 CSV 文件的路径。
        labels_path (str): 标签 CSV 文件的路径。

    返回:
        tuple: (X, y)，X 是特征 DataFrame，y 是标签 Series。
    """
    logger.info(f"开始加载数据 - 特征: {features_path}, 标签: {labels_path}")
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # 根据 'ID' 合并特征和标签
    df = pd.merge(features_df, labels_df, on='ID')

    # 找出不是特征的列
    non_feature_cols = ['ID', 'label']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    X = df[feature_cols]
    y = df['label']

    logger.info(f"数据加载完毕 - 样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    logger.info(f"标签分布情况:\n{y.value_counts()}")

    # 检查特征里有没有缺的值
    if X.isnull().values.any():
        raise ValueError("特征里发现有缺失值！")

    # 检查标签里有没有缺的值
    if y.isnull().values.any():
        raise ValueError("标签里发现有缺失值！")

    return X, y


def preprocess_data(X_train, X_val=None):
    """
    用 StandardScaler 标准化数据。

    参数:
        X_train: 训练集的特征。
        X_val: (可选) 验证集的特征。

    返回:
        tuple: 返回标准化后的训练数据、验证数据和 scaler 对象。
    """
    scaler = StandardScaler()
    # scaler 在训练集上学习并转换
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = None
    if X_val is not None:
        # 验证集只做转换（transform），不用再学习了
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    return X_train_scaled, scaler # 如果没有验证集，就只返回训练集和scaler


def calculate_youden_index(fpr, tpr, thresholds):
    """
    根据约登指数 (Youden's J statistic) 计算最佳阈值。

    参数:
        fpr (np.ndarray): 假正例率 (False Positive Rates)。
        tpr (np.ndarray): 真正例率 (True Positive Rates)。
        thresholds (np.ndarray): 对应的阈值。

    返回:
        tuple: (最佳阈值, 对应的最大约登指数)
    """
    youden_index = tpr - fpr
    # 找到约登指数最大的那个位置
    max_youden_idx = np.argmax(youden_index)
    best_threshold = thresholds[max_youden_idx]
    best_youden = youden_index[max_youden_idx]

    return best_threshold, best_youden


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    根据预测概率和给定的阈值，计算各种分类指标。

    参数:
        y_true: 真实的标签 (0或1)。
        y_pred_proba: 模型预测样本为正类(1)的概率。
        threshold (float, 可选): 分类阈值，默认0.5。

    返回:
        dict: 包含准确率、敏感性、特异性、PPV、NPV 和所用阈值的字典。
    """
    # # 确保 y_pred_proba 是一维的（只包含正类的概率）
    # if y_pred_proba.ndim > 1:
    #     logger.warning(f"预测概率 y_pred_proba 是 {y_pred_proba.shape} 形状的，应该是1维的。默认取第二列。")
    #     if y_pred_proba.shape[1] >= 2:
    #         y_pred_proba = y_pred_proba[:, 1]
    #     else: # 如果只有一列输出
    #          y_pred_proba = y_pred_proba.flatten()

    # 根据阈值得到预测的类别 (0或1)
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred) # 准确率
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性 (Sensitivity) / 召回率 (Recall) / 真阳性率 (TPR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性 (Specificity) / 真阴性率 (TNR)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0          # 阳性预测值 (PPV) / 精确率 (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # 阴性预测值 (NPV)

    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Threshold': threshold # 也记录一下用的哪个阈值
    }

    return metrics


def cross_validate_model(model_type, X, y, param_grid, cv=5, results_dir=None):
    """
    执行分层交叉验证来评估模型参数，然后用最佳参数在全部数据上训练最终模型，并保存结果。
    这个函数会用 ParameterGrid 来生成参数组合。
    对于ANN模型，它会利用验证集进行内部的早停。

    参数:
        model_type (str): 模型类型 ('LR', 'SVM', 'MLP')。
        X (pd.DataFrame): 特征数据。
        y (pd.Series): 标签数据。
        param_grid (dict): 定义模型参数搜索范围的字典。
        cv (int, 可选): 交叉验证的折数，默认是5。
        results_dir (str, 可选): 保存结果（模型、scaler、图等）的目录。默认不保存。

    返回:
        tuple: (best_model, best_params, cv_results_df)
               - best_model: 用最佳参数在所有数据上训练好的最终模型。
               - best_params: 找到的最佳参数组合 (字典)。
               - cv_results_df: 包含每一折、每种参数组合详细结果的 DataFrame。
    """
    logger.info(f"开始为 {model_type} 进行 {cv}-折交叉验证...")

    # 生成所有参数组合
    parameter_iterator = ParameterGrid(param_grid)
    all_param_combinations = list(parameter_iterator)
    if not all_param_combinations:
        logger.error("参数网格是空的！没法继续了。")
        return None, None, None
    logger.info(f"总共要评估 {len(all_param_combinations)} 种参数组合。")

    # --- 动态地初始化一个字典来存结果 ---
    cv_results = {
        'fold_idx': [], 'auc': [], 'best_threshold': [],
        'accuracy': [], 'sensitivity': [], 'specificity': [],
        'ppv': [], 'npv':[] 
    }
    # 找到所有可能出现的参数名
    all_possible_param_keys = set()
    for params in all_param_combinations:
        all_possible_param_keys.update(params.keys())
    # 把这些参数名也加到结果字典里，用来记录每折用了什么参数
    for param_name in sorted(list(all_possible_param_keys)):
        cv_results[param_name] = []
    # ---------------------------------------------

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) # 分层K折，保证每折标签比例差不多
    param_performance = [] # 用来存每种参数组合的平均性能 {'params': ..., 'mean_auc': ...}

    # --- 循环遍历每一种参数组合 ---
    for params in all_param_combinations:
        logger.info(f"正在评估参数: {params}")
        fold_aucs = []      # 存当前参数下，每一折的 AUC
        fold_thresholds = [] # 存当前参数下，每一折算出来的最佳阈值

        # --- 循环遍历交叉验证的每一折 ---
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # 划分训练集和验证集（在当前折内）
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # 预处理数据（scaler 只在当前折的训练集上 fit）
            X_train_scaled, X_val_scaled, fold_scaler = preprocess_data(X_train_fold, X_val_fold)

            # 准备创建模型所需的参数
            current_params = params.copy()
            if model_type.upper() == 'MLP':
                # MLP 模型需要知道输入特征的数量
                current_params['input_dim'] = X_train_scaled.shape[1]

            # 创建模型实例
            model = create_model(model_type, **current_params)

            # --- 训练模型 ---
            if model_type.upper() == 'MLP':
                # MLP 模型内部会用验证集 (X_val_scaled, y_val_fold) 来做早停
                logger.debug(f"  第 {fold_idx+1} 折: 训练 MLP (带内部验证)...")
                # 确保标签是 numpy array 给 MLP 的 fit 方法（如果它需要的话）
                model.fit(X_train_scaled, y_train_fold.to_numpy(),
                          X_val=X_val_scaled, y_val=y_val_fold.to_numpy())
            else:
                # 其他模型（LR, SVM）只在训练集上训练
                model.fit(X_train_scaled, y_train_fold)
            # -----------------

            # --- 在验证集上评估 ---
            y_val_proba = model.predict_proba(X_val_scaled) # 预测概率
            # # 确保 y_val_proba 是1维的，只含正类概率
            # if y_val_proba.ndim == 2:
            #      if y_val_proba.shape[1] >= 2:
            #          y_val_proba = y_val_proba[:, 1]
            #      else: # 处理只有一个输出列的情况
            #           y_val_proba = y_val_proba.flatten()

            # 计算 ROC 曲线和 AUC
            fpr, tpr, thresholds_roc = roc_curve(y_val_fold, y_val_proba)
            fold_auc = auc(fpr, tpr)

            # 计算这一折的最佳阈值（基于约登指数）
            best_threshold_fold, _ = calculate_youden_index(fpr, tpr, thresholds_roc)
            fold_thresholds.append(best_threshold_fold) # 存一下

            # 用算出来的这个最佳阈值来计算其他指标
            metrics = calculate_metrics(y_val_fold, y_val_proba, threshold=best_threshold_fold)

            # --- 保存这一折的结果 ---
            fold_aucs.append(fold_auc)
            cv_results['fold_idx'].append(fold_idx + 1) # 折数从1开始记
            cv_results['auc'].append(fold_auc)
            cv_results['best_threshold'].append(best_threshold_fold)
            cv_results['accuracy'].append(metrics['Accuracy'])
            cv_results['sensitivity'].append(metrics['Sensitivity'])
            cv_results['specificity'].append(metrics['Specificity'])
            cv_results['ppv'].append(metrics['PPV'])
            cv_results['npv'].append(metrics['NPV'])

            # 记录这折用的参数
            for param_name in all_possible_param_keys:
                cv_results[param_name].append(params.get(param_name)) # 用 get 安全一点，防止某个参数组合没这个键
            # -----------------------------------

            logger.info(f"  第 {fold_idx+1}/{cv} 折 - 验证集 AUC: {fold_auc:.4f}, 最佳阈值: {best_threshold_fold:.4f}, 准确率: {metrics['Accuracy']:.4f}")

        # --- 当前参数组合的所有折都跑完了 ---
        # 计算这个参数组合在所有折上的平均表现
        mean_auc = np.nanmean(fold_aucs) if fold_aucs else 0
        mean_threshold = np.nanmean(fold_thresholds) if fold_thresholds else 0.5 
        param_performance.append({'params': params, 'mean_auc': mean_auc, 'mean_threshold': mean_threshold})
        logger.info(f"参数 {params} - 在 {cv} 折上的平均验证集 AUC: {mean_auc:.4f}")
        # --------------------------------------------------

    # --- 找出平均验证 AUC 最高的参数组合 ---
    # 按 mean_auc 降序排，找到最好的那个
    best_performance = max(param_performance, key=lambda x: x['mean_auc'] if not np.isnan(x['mean_auc']) else -np.inf)

    best_params = best_performance['params']
    best_mean_auc = best_performance['mean_auc']
    # best_threshold_from_cv = best_performance['mean_threshold'] # 这是最佳参数组合在CV中的平均阈值
    logger.info(f"交叉验证结束。最佳平均验证 AUC: {best_mean_auc:.4f}，对应参数: {best_params}")

    # 把详细结果转成 DataFrame
    cv_results_df = pd.DataFrame(cv_results)

    # --- 用找到的最佳参数，在 *全部* 数据上训练最终模型 ---
    logger.info(f"开始用最佳参数在 *整个数据集* 上训练最终的 {model_type} 模型...")

    # 重新处理整个数据集（用一个新的 scaler，在所有数据上 fit）
    # 这是常见做法，假设测试集将来也会用类似方式处理
    X_scaled, final_scaler = preprocess_data(X)
    y_np = y.to_numpy() # 转成 numpy array 可能更通用

    # 准备最终模型的参数
    final_best_params = best_params.copy()
    if model_type.upper() == 'MLP':
        final_best_params['input_dim'] = X_scaled.shape[1] # 别忘了给最终的 MLP 也加上 input_dim

    # 创建最终模型
    best_model = create_model(model_type, **final_best_params)

    # 训练最终模型
    if model_type.upper() == 'MLP':
        # 对于最终的 MLP 模型训练，也需要一个小的内部验证集来配合早停机制
        logger.info("为最终的 MLP 模型训练创建一个内部验证集 (10%) 用于早停...")
        X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
            X_scaled, y_np, test_size=0.1, random_state=42, stratify=y_np # 分层抽样
        )
        best_model.fit(X_final_train, y_final_train, X_val=X_final_val, y_val=y_final_val)
    else:
        # 其他模型（LR, SVM）直接在全部缩放后的数据上训练
        best_model.fit(X_scaled, y_np)

    logger.info("最终模型训练完成。")

    # --- 在整个训练集上评估最终模型 ---
    train_proba = best_model.predict_proba(X_scaled)
    if train_proba.ndim == 2: # 确保是一维概率
         if train_proba.shape[1] >= 2:
             train_proba = train_proba[:, 1]
         else:
              train_proba = train_proba.flatten()

    fpr_train, tpr_train, thresholds_train = roc_curve(y_np, train_proba)
    train_auc = auc(fpr_train, tpr_train)

    # 在整个训练集的预测结果上，计算最终的最佳阈值
    # 这个阈值保存下来，用在测试集上
    final_best_threshold, _ = calculate_youden_index(fpr_train, tpr_train, thresholds_train)
    logger.info(f"在 *整个训练集* 预测结果上计算得到的最终阈值: {final_best_threshold:.4f}")

    # 用这个最终阈值，计算在整个训练集上的各项指标
    train_metrics = calculate_metrics(y_np, train_proba, threshold=final_best_threshold)

    logger.info(f"最终模型在 *整个训练集* 上的表现:")
    logger.info(f"  AUC: {train_auc:.4f}")
    logger.info(f"  使用阈值 {final_best_threshold:.4f} 时的指标:")
    for metric, value in train_metrics.items():
        if metric != 'Threshold':
            logger.info(f"    {metric}: {value:.4f}")


    os.makedirs(results_dir, exist_ok=True) # 确保目录存在
    base_filename = f"{model_type.lower()}"

    # 保存最终训练好的模型
    model_path = os.path.join(results_dir, f"{base_filename}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # 保存最终的 scaler
    scaler_path = os.path.join(results_dir, f"{base_filename}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(final_scaler, f)

    # 保存最终计算出的阈值
    threshold_path = os.path.join(results_dir, f"{base_filename}_threshold.pkl")
    with open(threshold_path, 'wb') as f:
        pickle.dump(final_best_threshold, f)

    # 保存详细的交叉验证结果
    cv_results_path = os.path.join(results_dir, f"{base_filename}_cv_results.csv")
    cv_results_df.to_csv(cv_results_path, index=False)

    # 保存训练集上的 ROC 曲线图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2,
                label=f'训练集 ROC 曲线 (AUC = {train_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title(f'{model_type} 模型 - 训练集 ROC 曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_save_path = os.path.join(results_dir, f"{base_filename}_train_roc_curve.png")
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    logger.info(f"最终模型、scaler、阈值、CV结果和ROC图已保存到 {results_dir} (使用固定文件名如 {base_filename}_*)")

    # 返回训练好的模型、最佳参数和CV结果
    return best_model, best_params, cv_results_df


def main():
    # --- 配置区域 ---
    config = {
        'features_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/radiomics/final_train_features.csv', # 特征文件
        'labels_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/train_labels.csv',       # 标签文件
        'results_dir': '/home/vipuser/Desktop/Classification/train_results_radiomics',                # 保存结果的目录
        'models_to_train': ['LR', 'SVM', 'MLP'],             # 要训练哪些模型
        'random_state': 42                                   # 全局随机种子，保证结果可复现
    }

    # 创建基础的结果目录
    base_results_dir = config['results_dir']
    os.makedirs(base_results_dir, exist_ok=True)

    # --- 定义不同模型的参数搜索范围 ---
    param_grids = {
        'LR': {
            # 'C': [0.01, 0.1, 1.0, 10.0], # 正则化强度的倒数
            # 'solver': ['liblinear'],    # 这个求解器比较适合小数据集，也能用L1/L2正则化
            # 'max_iter': [1000],         
            'C': [0.001], # 正则化强度的倒数
            'max_iter': [50],         
            'random_state': [config['random_state']]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0, 50.0], # 惩罚参数 C，范围扩大一点试试
            'kernel': ['rbf', 'linear'], # 尝试线性和高斯核
            'gamma': ['scale', 'auto'],  # gamma 参数主要影响 'rbf' 核
            'probability': [True],       # 必须设为 True 才能用 predict_proba 算概率
            'random_state': [config['random_state']]
        },
        'MLP': { 
            # 网络结构
            # 'hidden_dim1': [32, 64],     # 第一个隐藏层的大小
            # 'hidden_dim2': [16, 32],     # 第二个隐藏层的大小
            # 'dropout_rate': [0.3, 0.5],  # Dropout 比率
            'hidden_dim1': [32],     # 第一个隐藏层的大小
            'hidden_dim2': [16],     # 第二个隐藏层的大小
            'dropout_rate': [0.1],  # Dropout 比率
            # 优化器和正则化
            # 'learning_rate': [0.001, 0.005], # 学习率
            # 'weight_decay': [1e-5, 1e-4, 1e-3], # 权重衰减
            'learning_rate': [0.0007], # 学习率
            'weight_decay': [1e-5], # 权重衰减
            # 训练控制
            # 'batch_size': [32, 64],            # 批大小
            'batch_size': [64],            # 批大小
            'num_epochs': [150],              # 最大训练轮数
            'early_stopping_patience': [20],  # 早停的耐心值
            # 学习率调度器 (具体的 patience/T_max/gamma 在 ANNModel 内部设置)
            # 'lr_scheduler_type': ['plateau', 'cosine'], # 尝试两种学习率衰减策略
            'lr_scheduler_type': ['cosine'], # 尝试两种学习率衰减策略
            # 其他
            'random_state': [config['random_state']]
        }
    }

    # --- 加载数据 ---
    logger.info("准备加载数据...")
    X, y = load_data(config['features_path'], config['labels_path'])

    # --- 依次训练和评估指定的模型 ---
    all_results = {} # 用来存每个模型的结果
    for model_type in config['models_to_train']:
        if model_type not in param_grids:
            logger.warning(f"模型 '{model_type}' 没有定义参数网格，跳过训练。")
            continue

        logger.info(f"===== 开始训练模型: {model_type} =====")

        # 为这个模型创建一个专门的子目录来存结果
        model_results_dir = os.path.join(base_results_dir, model_type)
        os.makedirs(model_results_dir, exist_ok=True)

        # 运行交叉验证、参数搜索、最终模型训练和保存
        _, best_params, cv_results_df = cross_validate_model(
            model_type=model_type,
            X=X,
            y=y,
            param_grid=param_grids[model_type],
            cv=5, # 用5折交叉验证
            results_dir=model_results_dir # 把结果存到模型自己的目录里
        )

        all_results[model_type] = {
            # 'best_model': best_model, # 如果后面还需要用模型对象可以取消注释
            'best_params': best_params,
            'cv_results_df': cv_results_df
        }
        # 从返回的 DataFrame 里算一下平均 CV AUC (注意处理可能的 NaN)
        avg_cv_auc = cv_results_df['auc'].mean() if not cv_results_df['auc'].isnull().all() else np.nan

        logger.info(f"===== {model_type} 训练总结 =====")
        logger.info(f"  最佳参数: {best_params}")
        logger.info(f"  平均交叉验证 AUC: {avg_cv_auc:.4f}")
        logger.info(f"  结果已保存在: {model_results_dir}")

        logger.info(f"===== {model_type} 模型训练结束 =====")


    # --- 生成最终的总结报告 ---
    summary_data = []
    for model_type, results in all_results.items():
        # 从存下来的 DataFrame 里安全地重新计算平均 CV AUC
        cv_df = results['cv_results_df']
        avg_cv_auc = cv_df['auc'].mean() if not cv_df['auc'].isnull().all() else np.nan

        summary_data.append({
            '模型类型': model_type,
            '平均CV AUC': f"{avg_cv_auc:.4f}",
            '最佳参数': str(results['best_params']) 
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(base_results_dir, "overall_training_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"所有模型训练的总结报告已保存到: {summary_path}")
    print("\n--- 所有模型训练总结 ---")
    print(summary_df.to_string(index=False))
    print("-------------------------")

    logger.info("运行结束。")


if __name__ == "__main__":
    main()