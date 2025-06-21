# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
# from sklearn.metrics import roc_auc_score
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR
# import copy
# import logging # Using logging for info/warnings

# # Basic logging setup (optional, but good practice)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class LogisticRegressionModel:
#     """
#     Wrapper class for Logistic Regression model.
#     """
#     def __init__(self, C=1.0, max_iter=1000, random_state=42, **kwargs):
#         """
#         Initialize Logistic Regression model.
        
#         Args:
#             C (float): Inverse of regularization strength
#             max_iter (int): Maximum number of iterations
#             random_state (int): Random seed for reproducibility
#             **kwargs: Additional parameters for LogisticRegression
#         """
#         self.model = LogisticRegression(
#             C=C,
#             max_iter=max_iter,
#             random_state=random_state,
#             **kwargs
#         )
#         self.name = "Logistic Regression"
#         self.params = {'C': C, 'max_iter': max_iter, 'random_state': random_state, **kwargs}
        
#     def fit(self, X, y):
#         """Train the model"""
#         self.model.fit(X, y)
#         return self
        
#     def predict(self, X):
#         """Make binary predictions"""
#         return self.model.predict(X)
    
#     def predict_proba(self, X):
#         """Get probability estimates for positive class"""
#         return self.model.predict_proba(X)[:, 1]


# class SVMModel:
#     """
#     Wrapper class for Support Vector Machine model.
#     """
#     def __init__(self, C=1.0, kernel='rbf', probability=True, random_state=42, **kwargs):
#         """
#         Initialize SVM model.
        
#         Args:
#             C (float): Regularization parameter
#             kernel (str): Kernel type (rbf, linear, poly, sigmoid)
#             probability (bool): Enable probability estimates
#             random_state (int): Random seed for reproducibility
#             **kwargs: Additional parameters for SVC
#         """
#         self.model = SVC(
#             C=C,
#             kernel=kernel,
#             probability=probability,
#             random_state=random_state,
#             **kwargs
#         )
#         self.name = "Support Vector Machine"
#         self.params = {'C': C, 'kernel': kernel, 'probability': probability, 
#                        'random_state': random_state, **kwargs}
        
#     def fit(self, X, y):
#         """Train the model"""
#         self.model.fit(X, y)
#         return self
        
#     def predict(self, X):
#         """Make binary predictions"""
#         return self.model.predict(X)
    
#     def predict_proba(self, X):
#         """Get probability estimates for positive class"""
#         return self.model.predict_proba(X)[:, 1]
    
# class ANN(nn.Module):
#     """
#     双隐藏层神经网络架构 (保持不变)
#     """
#     def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.5):
#         super(ANN, self).__init__()
#         # 第一隐藏层
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim1) # 第一层批量归一化
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout_rate)

#         # 第二隐藏层
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim2) # 第二层批量归一化
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout_rate)

#         # 输出层
#         self.fc3 = nn.Linear(hidden_dim2, 1) # 输出维度为1（二分类）

#     def forward(self, x):
#         # 第一隐藏层
#         x = self.fc1(x)
#         # 批量归一化需要至少2个样本才能计算统计量，评估时可能输入单个样本
#         # if x.shape[0] > 1:
#         #     x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)

#         # 第二隐藏层
#         x = self.fc2(x)
#         if x.shape[0] > 1:
#             x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)

#         # 输出层：线性变换 + Sigmoid激活（用于二分类概率）
#         # x = torch.sigmoid(self.fc3(x))
#         x = self.fc3(x)
#         return x


# class ANNModel:
#     """
#     包装ANN模型的类，提供类似scikit-learn的接口。
#     强制使用验证集AUC进行早停和模型选择。
#     支持特定的学习率调度器。
#     """
#     def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.5,
#                  learning_rate=0.001, batch_size=32, num_epochs=100,
#                  weight_decay=1e-4,
#                  early_stopping_patience=10,
#                  # --- 学习率调度器参数 ---
#                  lr_scheduler_type='plateau', # 'plateau', 'cosine', 'exponential'
#                  lr_scheduler_patience=5,   # For 'plateau'
#                  lr_scheduler_T_max=50,     # For 'cosine'
#                  lr_scheduler_gamma=0.95,   # For 'exponential'
#                  # --------------------------
#                  random_state=42, device=None, **kwargs):
#         """
#         初始化ANN模型

#         参数：
#             input_dim (int): 输入特征数量
#             hidden_dim1 (int): 第一个隐藏层大小
#             hidden_dim2 (int): 第二个隐藏层大小
#             dropout_rate (float): Dropout概率
#             learning_rate (float): 优化器初始学习率
#             batch_size (int): 训练批次大小
#             num_epochs (int): 最大训练轮数
#             weight_decay (float): 优化器的权重衰减（L2正则化）系数
#             early_stopping_patience (int): 早停的耐心轮数（基于验证集AUC）
#             lr_scheduler_type (str): 使用的学习率调度器类型: 'plateau', 'cosine', 'exponential'
#             lr_scheduler_patience (int): ReduceLROnPlateau的耐心值 (仅当 type='plateau')
#             lr_scheduler_T_max (int): CosineAnnealingLR的最大迭代次数 (仅当 type='cosine')
#             lr_scheduler_gamma (float): ExponentialLR的乘法因子 (仅当 type='exponential')
#             random_state (int): 随机种子，用于结果可复现
#             device (str): 使用的设备（'cuda'或'cpu'），None则自动检测
#             **kwargs: 其他参数（当前未使用）
#         """
#         # 设置随机种子
#         torch.manual_seed(random_state)
#         np.random.seed(random_state)

#         self.input_dim = input_dim
#         self.hidden_dim1 = hidden_dim1
#         self.hidden_dim2 = hidden_dim2
#         self.dropout_rate = dropout_rate
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.weight_decay = weight_decay
#         self.early_stopping_patience = early_stopping_patience

#         # --- 学习率调度器设置 ---
#         valid_schedulers = ['plateau', 'cosine', 'exponential']
#         if lr_scheduler_type.lower() not in valid_schedulers:
#             raise ValueError(f"lr_scheduler_type must be one of {valid_schedulers}")
#         self.lr_scheduler_type = lr_scheduler_type.lower()
#         self.lr_scheduler_patience = lr_scheduler_patience
#         self.lr_scheduler_T_max = lr_scheduler_T_max
#         self.lr_scheduler_gamma = lr_scheduler_gamma
#         # --------------------------

#         self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.name = f"ANN (BN, EarlyStop:AUC, Scheduler:{self.lr_scheduler_type})"

#         # 创建神经网络
#         self.model = ANN(
#             input_dim=input_dim,
#             hidden_dim1=hidden_dim1,
#             hidden_dim2=hidden_dim2,
#             dropout_rate=dropout_rate
#         ).to(self.device)

#         self.best_model_state = None
#         # 保留 train_loss, val_loss, val_auc 和 lr 用于记录
#         self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}

#         # 存储所有有效的初始化参数
#         self.params = {
#             'input_dim': input_dim, 'hidden_dim1': hidden_dim1, 'hidden_dim2': hidden_dim2,
#             'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'batch_size': batch_size,
#             'num_epochs': num_epochs, 'weight_decay': weight_decay,
#             'early_stopping_patience': early_stopping_patience,
#             'lr_scheduler_type': self.lr_scheduler_type,
#             'lr_scheduler_patience': self.lr_scheduler_patience,
#             'lr_scheduler_T_max': self.lr_scheduler_T_max,
#             'lr_scheduler_gamma': self.lr_scheduler_gamma,
#             'random_state': random_state,
#             **kwargs # 包含任何未明确列出的kwargs
#         }

#     def fit(self, X, y, X_val, y_val): # 强制要求 X_val, y_val
#         """
#         训练模型。强制使用验证集AUC进行早停和模型选择。

#         重要提示：在调用此方法之前，请确保对输入特征 X 和 X_val 进行了适当的缩放。

#         参数:
#             X (numpy.ndarray or pandas.DataFrame): 训练特征
#             y (numpy.ndarray or pandas.Series): 训练标签 (0或1)
#             X_val (numpy.ndarray or pandas.DataFrame): 验证特征 (必需)
#             y_val (numpy.ndarray or pandas.Series): 验证标签 (必需)

#         返回:
#             self: 训练好的模型
#         """
#         # --- 强制验证集检查 ---
#         if X_val is None or y_val is None:
#             raise ValueError("Validation data (X_val, y_val) must be provided for fitting this model.")

#         # --- 数据预处理 ---
#         if isinstance(X, pd.DataFrame): X = X.values
#         if isinstance(y, pd.Series): y = y.values
#         X_tensor = torch.FloatTensor(X).to(self.device)
#         y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
#         dataset = TensorDataset(X_tensor, y_tensor)
#         loader = DataLoader(dataset, batch_size=int(self.batch_size), shuffle=True)

#         # 处理验证集
#         if isinstance(X_val, pd.DataFrame): X_val = X_val.values
#         if isinstance(y_val, pd.Series): y_val = y_val.values
#         X_val_tensor = torch.FloatTensor(X_val).to(self.device)
#         y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
#         y_val_numpy = y_val_tensor.cpu().numpy().flatten() # 用于计算AUC
#         logger.info(f"使用验证集进行早停和模型选择 (指标: AUC)。验证集大小: {len(X_val_tensor)}")

#         # 检查验证集标签是否有效
#         if len(np.unique(y_val_numpy)) < 2:
#             raise ValueError("Validation labels (y_val) must contain at least two classes to calculate AUC.")


#         # --- 模型设置 ---
#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#         criterion = nn.BCEWithLogitsLoss()

#         # --- 实例化选择的学习率调度器 ---
#         if self.lr_scheduler_type == 'plateau':
#             scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, # AUC 越高越好
#                                           patience=self.lr_scheduler_patience, verbose=True)
#         elif self.lr_scheduler_type == 'cosine':
#             scheduler = CosineAnnealingLR(optimizer, T_max=self.lr_scheduler_T_max, eta_min=0, verbose=False) # verbose=False避免过多输出
#         elif self.lr_scheduler_type == 'exponential':
#             scheduler = ExponentialLR(optimizer, gamma=self.lr_scheduler_gamma, verbose=False)
#         else:
#             # 这个应该不会发生，因为在 __init__ 中检查过了
#              raise ValueError(f"内部错误：无效的调度器类型 {self.lr_scheduler_type}")


#         # --- 训练循环 ---
#         best_val_auc = 0.0 # AUC 越高越好
#         patience_counter = 0
#         self.best_model_state = None
#         self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []} # 重置 history

#         logger.info(f"开始训练，设备: {self.device}, 最大轮数: {self.num_epochs}, 早停耐心: {self.early_stopping_patience}")

#         for epoch in range(self.num_epochs):
#             # --- 训练阶段 ---
#             self.model.train()
#             epoch_train_loss = 0
#             for inputs, targets in loader:
#                 optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_train_loss += loss.item()
#             avg_train_loss = epoch_train_loss / len(loader)
#             self.history['train_loss'].append(avg_train_loss)
#             current_lr = optimizer.param_groups[0]['lr']
#             self.history['lr'].append(current_lr)

#             # --- 验证阶段 ---
#             epoch_val_loss = None
#             epoch_val_auc = None
#             self.model.eval()
#             with torch.no_grad():
#                 val_outputs = self.model(X_val_tensor)
#                 # 计算验证损失 (用于记录)
#                 epoch_val_loss = criterion(val_outputs, y_val_tensor).item()
#                 self.history['val_loss'].append(epoch_val_loss)

#                 # 计算验证集 AUC (用于早停和模型选择)
#                 val_outputs_numpy = val_outputs.cpu().numpy().flatten()
#                 try:
#                     epoch_val_auc = roc_auc_score(y_val_numpy, val_outputs_numpy)
#                     self.history['val_auc'].append(epoch_val_auc)
#                 except ValueError as e:
#                     # 这通常不应该发生，因为我们在开始时检查了y_val
#                     logger.warning(f"警告：轮次 {epoch+1} 计算 AUC 时出错: {e}。将此轮 AUC 设为 0。")
#                     epoch_val_auc = 0.0 # 或 None? 设为0更安全，避免因None跳过比较
#                     self.history['val_auc'].append(epoch_val_auc)

#             # 打印日志
#             log_msg = (f"轮次 {epoch+1}/{self.num_epochs} - "
#                        f"训练损失: {avg_train_loss:.4f} - "
#                        f"验证损失: {epoch_val_loss:.4f} - "
#                        f"验证 AUC: {epoch_val_auc:.4f} - "
#                        f"学习率: {current_lr:.6f}")
#             logger.info(log_msg)

#             # --- 基于验证 AUC 的早停和模型保存 ---
#             if epoch_val_auc > best_val_auc:
#                 best_val_auc = epoch_val_auc
#                 self.best_model_state = copy.deepcopy(self.model.state_dict())
#                 patience_counter = 0
#                 # logger.info(f"  找到新的最佳模型，验证 AUC: {best_val_auc:.4f}")
#             else:
#                 patience_counter += 1
#                 if patience_counter >= self.early_stopping_patience:
#                     logger.info(f"验证 AUC 在 {self.early_stopping_patience} 轮内未改善，触发早停！")
#                     break

#             # --- 学习率调度器步骤 ---
#             if self.lr_scheduler_type == 'plateau':
#                 scheduler.step(epoch_val_auc) # Plateau 需要监控指标
#             else:
#                 scheduler.step() # Cosine 和 Exponential 按轮次更新

#         # --- 训练结束 ---
#         final_metric_str = f"AUC: {best_val_auc:.4f}"
#         if self.best_model_state is not None:
#             logger.info(f"训练结束。加载验证 AUC 最优的模型 ({final_metric_str})。")
#             self.model.load_state_dict(self.best_model_state)
#         else:
#              # 如果从未找到更好的模型（例如patience=0或第一轮最好）
#              logger.info(f"训练结束。使用最后一轮的模型状态 (未找到更优模型或早停未触发)。最佳验证 {final_metric_str}")

#         return self

#     def predict(self, X):
#         """
#         进行二分类预测 (0 或 1)。
#         """
#         proba = self.predict_proba(X)
#         return (proba >= 0.5).astype(int)

#     def predict_proba(self, X):
#         """
#         获取正类的概率估计 (一维数组)。
#         """
#         self.model.eval()
#         if isinstance(X, pd.DataFrame): X = X.values
#         # 确保输入是二维的
#         if X.ndim == 1:
#              X = X.reshape(1, -1)
#         elif X.ndim > 2:
#              raise ValueError(f"输入 X 的维度应为 1 或 2，但得到 {X.ndim}")

#         X_tensor = torch.FloatTensor(X).to(self.device)
#         with torch.no_grad():
#             logits = self.model(X_tensor) # 获取 logits
#             y_proba = torch.sigmoid(logits).cpu().numpy() # 应用 sigmoid 得到概率

#         return y_proba.reshape(-1) # 返回一维数组

#     def get_params(self, deep=True):
#         """获取模型参数，用于兼容 scikit-learn 工具"""
#         return self.params

#     def set_params(self, **params):
#         """设置模型参数，用于兼容 scikit-learn 工具"""
#         for key, value in params.items():
#             # 更新存储的参数字典
#             self.params[key] = value
#             # 同时尝试更新类实例的属性 (如果存在)
#             if hasattr(self, key):
#                  # 特殊处理调度器类型
#                  if key == 'lr_scheduler_type':
#                       valid_schedulers = ['plateau', 'cosine', 'exponential']
#                       if value.lower() not in valid_schedulers:
#                           raise ValueError(f"lr_scheduler_type must be one of {valid_schedulers}")
#                       self.lr_scheduler_type = value.lower()
#                       self.name = f"ANN (BN, EarlyStop:AUC, Scheduler:{self.lr_scheduler_type})" # 更新名称
#                  else:
#                       setattr(self, key, value)
#             # 模型结构参数更改需要重新创建模型
#             if key in ['input_dim', 'hidden_dim1', 'hidden_dim2', 'dropout_rate']:
#                 logger.info(f"参数 '{key}' 已更改，正在重新创建模型结构...")
#                 self.model = ANN(
#                     input_dim=self.params['input_dim'],
#                     hidden_dim1=self.params['hidden_dim1'],
#                     hidden_dim2=self.params['hidden_dim2'],
#                     dropout_rate=self.params['dropout_rate']
#                 ).to(self.device)
#         return self

# # --- 示例用法 ---
# if __name__ == '__main__':
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.datasets import make_classification
#     from sklearn.metrics import accuracy_score, roc_auc_score

#     # 1. 创建模拟数据 (小数据集)
#     X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
#                                n_redundant=5, n_clusters_per_class=1, random_state=42)
#     print(f"原始数据形状: X={X.shape}, y={y.shape}")

#     # 2. 划分训练集、验证集、测试集
#     X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, # 0.25 * 0.8 = 0.2 -> 60% train, 20% val, 20% test
#                                                     random_state=42, stratify=y_train_val)

#     print(f"训练集大小: {len(X_train)}")
#     print(f"验证集大小: {len(X_val)}")
#     print(f"测试集大小: {len(X_test)}")

#     # 3. 特征缩放 (非常重要!)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val) # 使用训练集的 scaler 转换验证集
#     X_test_scaled = scaler.transform(X_test)  # 使用训练集的 scaler 转换测试集

#     # 4. 初始化和训练模型
#     input_dim = X_train_scaled.shape[1]
#     model = ANNModel(
#         input_dim=input_dim,
#         hidden_dim1=32,      # 减少隐藏层大小，适应小数据集
#         hidden_dim2=16,
#         dropout_rate=0.3,    # 调整 Dropout
#         learning_rate=0.005,
#         batch_size=16,       # 较小的批次大小可能在小数据集上效果更好
#         num_epochs=200,      # 增加最大轮数，让早停决定何时停止
#         weight_decay=0.001,  # 尝试 L2 正则化
#         early_stopping_patience=15,
#         lr_scheduler_patience=7,
#         random_state=42
#     )

#     print("\n--- 开始训练 ---")
#     # 传入验证集进行训练
#     model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val)
#     print("--- 训练完成 ---")

#     # 5. 在测试集上评估
#     print("\n--- 在测试集上评估 ---")
#     y_pred_proba = model.predict_proba(X_test_scaled)
#     y_pred = model.predict(X_test_scaled) # 等价于 (y_pred_proba >= 0.5).astype(int)

#     accuracy = accuracy_score(y_test, y_pred)
#     try:
#         auc = roc_auc_score(y_test, y_pred_proba)
#         print(f"测试集准确率 (Accuracy): {accuracy:.4f}")
#         print(f"测试集AUC: {auc:.4f}")
#     except ValueError:
#         # 如果测试集标签只有一类，AUC无法计算
#         print(f"测试集准确率 (Accuracy): {accuracy:.4f}")
#         print("测试集标签只有一类，无法计算AUC。")

#     # 打印模型使用的参数
#     # print("\n模型使用的参数:")
#     # print(model.get_params())


# def create_model(model_type, **kwargs):
#     """
#     Factory function to create models of specified type.
    
#     Args:
#         model_type (str): Type of model ('LR', 'SVM', or 'ANN')
#         **kwargs: Model-specific parameters
        
#     Returns:
#         Model instance with scikit-learn compatible interface
#     """
#     if model_type.upper() == 'LR':
#         return LogisticRegressionModel(**kwargs)
#     elif model_type.upper() == 'SVM':
#         return SVMModel(**kwargs)
#     elif model_type.upper() == 'ANN':
#         # Ensure input_dim is provided for ANN
#         if 'input_dim' not in kwargs:
#             raise ValueError("input_dim must be provided for ANN model")
#         return ANNModel(**kwargs)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")





















import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold # 在这个文件里没用到，可以注释掉
# from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score # 在这个文件里没用到，可以注释掉
from sklearn.metrics import roc_auc_score # 用来计算 AUC
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR
import copy
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogisticRegressionModel:
    """
    逻辑回归模型的包装类。
    """
    def __init__(self, C=1.0, max_iter=1000, random_state=42, **kwargs):
        """
        初始化逻辑回归模型。

        参数:
            C (float): 正则化强度的倒数，越小表示正则化越强。
            max_iter (int): 求解器最大迭代次数。
            random_state (int): 随机种子，保证结果可重现。
            **kwargs: 可以传给 scikit-learn LogisticRegression 的其他参数。
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.params = {'C': C, 'max_iter': max_iter, 'random_state': random_state, **kwargs}

    def fit(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        return self # 返回自身，方便链式调用

    def predict(self, X):
        """进行二分类预测 (返回 0 或 1)"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """获取正类 (类别1) 的概率估计 (返回一维数组)"""
        # scikit-learn 的 predict_proba 返回 N行 * K类 的数组
        # 对于二分类，我们通常只需要正类 (第二列) 的概率
        return self.model.predict_proba(X)[:, 1]


class SVMModel:
    """
    支持向量机 (SVM) 模型的包装类。
    """
    def __init__(self, C=1.0, kernel='rbf', probability=True, random_state=42, **kwargs):
        """
        初始化 SVM 模型。

        参数:
            C (float): 正则化参数。
            kernel (str): 核函数类型 ('rbf', 'linear', 'poly', 'sigmoid')。
            probability (bool): 是否启用概率估计 (必须为 True 才能用 predict_proba)。
            random_state (int): 随机种子。
            **kwargs: 可以传给 scikit-learn SVC 的其他参数。
        """

        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=probability, # 必须为 True 才能预测概率
            random_state=random_state,
            **kwargs
        )
        self.params = {'C': C, 'kernel': kernel, 'probability': probability,
                       'random_state': random_state, **kwargs}

    def fit(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """进行二分类预测 (返回 0 或 1)"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """获取正类 (类别1) 的概率估计 (返回一维数组)"""
        return self.model.predict_proba(X)[:, 1]

class ANN(nn.Module):
    """
    双隐藏层神经网络架构 (保持不变)
    """
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.5):
        super(ANN, self).__init__()
        # 第一隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1) # 第一层批量归一化
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第二隐藏层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2) # 第二层批量归一化
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # 输出层
        self.fc3 = nn.Linear(hidden_dim2, 1) # 输出维度为1（二分类）

    def forward(self, x):
        # 第一隐藏层
        x = self.fc1(x)
        # if x.shape[0] > 1:
        #     x = self.bn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 第二隐藏层
        x = self.fc2(x)
        # if x.shape[0] > 1:
        #     x = self.bn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 输出层：线性变换 + Sigmoid激活（用于二分类概率）
        # x = torch.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x

class ANNModel:
    """
    包装 PyTorch ANN 模型的类，提供类似 scikit-learn 的接口 (fit, predict, predict_proba)。
    使用验证集和 AUC 指标来进行早停和模型选择。
    """
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.5,
                 learning_rate=0.001, batch_size=32, num_epochs=100,
                 weight_decay=1e-4, # L2 正则化系数
                 early_stopping_patience=10, # 早停的耐心值 (基于验证集 AUC)
                 # --- 学习率调度器相关参数 ---
                 lr_scheduler_type='plateau', # 类型: 'plateau', 'cosine', 'exponential'
                 lr_scheduler_patience=5,   # 用于 'plateau' 调度器
                 lr_scheduler_T_max=50,     # 用于 'cosine' 调度器 (一个周期的 epoch 数)
                 lr_scheduler_gamma=0.95,   # 用于 'exponential' 调度器 (衰减因子)
                 # --------------------------
                 random_state=42, device=None, **kwargs): # kwargs 允许接收但不使用额外的参数
        """
        初始化 ANN 包装模型。

        参数:
            input_dim (int): 输入特征的数量。
            hidden_dim1 (int): 第一个隐藏层神经元数量。
            hidden_dim2 (int): 第二个隐藏层神经元数量。
            dropout_rate (float): Dropout 比率 (0 到 1)。
            learning_rate (float): Adam 优化器的初始学习率。
            batch_size (int): 训练时每个批次的大小。
            num_epochs (int): 最大的训练轮数 (可能因早停提前结束)。
            weight_decay (float): Adam 优化器的权重衰减系数 (L2 正则化)。
            early_stopping_patience (int): 如果验证集 AUC 连续这么多轮没有提升，就停止训练。
            lr_scheduler_type (str): 用哪种学习率调整策略 ('plateau', 'cosine', 'exponential')。
            lr_scheduler_patience (int): 用于 'plateau' 策略，AUC 多少轮没提升就降低学习率。
            lr_scheduler_T_max (int): 用于 'cosine' 策略，学习率变化周期的长度 (epoch 数)。
            lr_scheduler_gamma (float): 用于 'exponential' 策略，每轮学习率乘以的因子。
            random_state (int): 随机种子，用于 PyTorch 和 NumPy。
            device (str): 指定运行设备 ('cuda' 或 'cpu')，默认 None 会自动检测 GPU。
            **kwargs: 额外的参数，目前这个类不会用到它们。
        """
        # 设置随机种子保证可复现性
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience

        # --- 处理学习率调度器类型 ---
        self.lr_scheduler_type = lr_scheduler_type.lower()
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_T_max = lr_scheduler_T_max
        self.lr_scheduler_gamma = lr_scheduler_gamma
        # --------------------------

        # 自动选择设备 (GPU 优先)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建底层的 PyTorch 神经网络模型
        self.model = ANN(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout_rate=dropout_rate
        ).to(self.device) # 把模型放到指定的设备上

        self.best_model_state = None # 用来保存验证集上表现最好的模型状态
        # 用字典记录训练过程中的损失、AUC 和学习率
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}

        self.params = {
            'input_dim': input_dim, 'hidden_dim1': hidden_dim1, 'hidden_dim2': hidden_dim2,
            'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'batch_size': batch_size,
            'num_epochs': num_epochs, 'weight_decay': weight_decay,
            'early_stopping_patience': early_stopping_patience,
            'lr_scheduler_type': self.lr_scheduler_type,
            'lr_scheduler_patience': self.lr_scheduler_patience,
            'lr_scheduler_T_max': self.lr_scheduler_T_max,
            'lr_scheduler_gamma': self.lr_scheduler_gamma,
            'random_state': random_state,
            **kwargs # 也把额外的kwargs存起来
        }


    def fit(self, X, y, X_val, y_val):
        """
        训练模型。提供验证集 (X_val, y_val) 用于早停和选择最佳模型。

        参数:
            X (numpy.ndarray or pandas.DataFrame): 训练集特征。
            y (numpy.ndarray or pandas.Series): 训练集标签 (0 或 1)。
            X_val (numpy.ndarray or pandas.DataFrame): 验证集特征 (必需)。
            y_val (numpy.ndarray or pandas.Series): 验证集标签 (必需)。

        返回:
            self: 训练完成的模型实例。
        """

        # --- 数据准备：转换为 Tensor，创建 DataLoader ---
        # 训练集
        if isinstance(X, pd.DataFrame): X = X.values 
        if isinstance(y, pd.Series): y = y.values  
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device) # 标签需要是 [N, 1] 的形状
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=int(self.batch_size), shuffle=True, drop_last=True) # 打乱数据

        # 验证集 (不需要 DataLoader，因为通常一次性评估整个验证集)
        if isinstance(X_val, pd.DataFrame): X_val = X_val.values
        if isinstance(y_val, pd.Series): y_val = y_val.values
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        # 把验证集标签转回 numpy 格式，方便后面用 scikit-learn 计算 AUC
        y_val_numpy = y_val # 不需要 .cpu().numpy().flatten()，因为已经是 numpy 了

        logger.info(f"使用验证集进行早停和模型选择 (指标: AUC)。验证集大小: {len(X_val_tensor)}")

        # --- 模型、优化器、损失函数、学习率调度器 设置 ---
        # Adam 优化器，加入权重衰减 (L2 正则化)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # 带 Sigmoid 的二分类交叉熵损失，这个函数内部会处理 logits，更稳定
        criterion = nn.BCEWithLogitsLoss()

        # --- 根据选择的类型，实例化学习率调度器 ---
        scheduler = None # 先初始化为 None
        if self.lr_scheduler_type == 'plateau':
            # 当验证 AUC 不再提升时，降低学习率
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, # AUC 是越大越好 (mode='max')
                                          patience=self.lr_scheduler_patience, verbose=True) # verbose=True 会打印学习率变化信息
        elif self.lr_scheduler_type == 'cosine':
            # 余弦退火学习率，周期性变化
            scheduler = CosineAnnealingLR(optimizer, T_max=self.lr_scheduler_T_max, eta_min=0) # eta_min=0 表示最低降到0
        elif self.lr_scheduler_type == 'exponential':
            # 指数衰减学习率
            scheduler = ExponentialLR(optimizer, gamma=self.lr_scheduler_gamma)

        # --- 训练循环 ---
        best_val_auc = 0.0 # 初始化最佳验证 AUC 为 0 (因为 AUC >= 0)
        patience_counter = 0 # 记录验证 AUC 没有提升的轮数
        self.best_model_state = None # 清空上次训练可能留下的状态
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []} # 重置训练历史记录

        logger.info(f"开始训练，使用设备: {self.device}, 最多训练 {self.num_epochs} 轮, 早停耐心: {self.early_stopping_patience} 轮")

        for epoch in range(self.num_epochs):
            # --- 训练阶段 ---
            self.model.train() # 设置模型为训练模式 (启用 Dropout, BatchNorm 使用批次统计)
            epoch_train_loss = 0 # 记录当前轮的总训练损失
            for inputs, targets in loader: # 从 DataLoader 获取小批量数据
                optimizer.zero_grad() # 清空上一批的梯度
                outputs = self.model(inputs) # 前向传播，得到 logits
                loss = criterion(outputs, targets) # 计算损失
                loss.backward() # 反向传播，计算梯度
                optimizer.step() # 更新模型参数
                epoch_train_loss += loss.item() # 累加损失值
            avg_train_loss = epoch_train_loss / len(loader) # 计算平均训练损失
            self.history['train_loss'].append(avg_train_loss) # 记录
            current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
            self.history['lr'].append(current_lr) # 记录

            # --- 验证阶段 ---
            epoch_val_loss = None # 当前轮的验证损失
            epoch_val_auc = None  # 当前轮的验证 AUC
            self.model.eval() # 设置模型为评估模式 (禁用 Dropout, BatchNorm 使用运行统计)
            with torch.no_grad(): # 评估时不需要计算梯度
                val_outputs = self.model(X_val_tensor) # 对整个验证集进行预测，得到 logits
                # 计算验证损失 (可选，主要用于观察)
                epoch_val_loss = criterion(val_outputs, y_val_tensor).item()
                self.history['val_loss'].append(epoch_val_loss)

                # 计算验证集 AUC (这是早停和模型选择的关键指标)
                # 先将 logits 通过 sigmoid 转换成概率
                val_probs_numpy = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                epoch_val_auc = roc_auc_score(y_val_numpy, val_probs_numpy)
                self.history['val_auc'].append(epoch_val_auc)

            # 打印当前轮的日志信息
            log_msg = (f"轮次 {epoch+1}/{self.num_epochs} - "
                       f"训练损失: {avg_train_loss:.4f} - "
                       f"验证损失: {epoch_val_loss:.4f} - "
                       f"验证 AUC: {epoch_val_auc:.4f} - "
                       f"学习率: {current_lr:.6f}")
            logger.info(log_msg)

            # --- 基于验证 AUC 进行早停判断 和 保存最佳模型 ---
            if epoch_val_auc > best_val_auc: # 如果当前轮的 AUC 比之前记录的最好 AUC 要高
                best_val_auc = epoch_val_auc # 更新最好的 AUC
                # 深度拷贝当前模型的参数状态，保存起来
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0 # 重置耐心计数器
            else: # 如果当前轮的 AUC 没有变得更好
                patience_counter += 1 # 增加耐心计数器
                if patience_counter >= self.early_stopping_patience: # 如果耐心耗尽
                    logger.info(f"验证 AUC 在最近 {self.early_stopping_patience} 轮没有提升，触发早停！")
                    break # 跳出训练循环

            # --- 更新学习率 (根据选择的调度器) ---
            if scheduler: # 确保调度器已创建
                if self.lr_scheduler_type == 'plateau':
                    scheduler.step(epoch_val_auc) # Plateau 调度器需要传入监控的指标 (AUC)
                else:
                    # Cosine 和 Exponential 调度器是按轮次更新的
                    scheduler.step()

        # --- 训练循环结束 ---
        final_metric_str = f"AUC: {best_val_auc:.4f}" # 记录最终的最佳验证 AUC
        # 如果在训练过程中保存过最佳模型状态 (即至少有一轮比初始值好)
        if self.best_model_state is not None:
            logger.info(f"训练结束。加载验证集 AUC 最优的模型状态 ({final_metric_str})。")
            # 将模型参数加载回最佳状态
            self.model.load_state_dict(self.best_model_state)
        else:
             # 就使用训练结束时的最后一轮模型状态
             logger.info(f"训练结束。使用最后一轮的模型状态 (可能未找到更优模型或早停未触发)。记录的最佳验证指标为 {final_metric_str}")
             # 这种情况下，模型已经是最后一轮的状态了，不需要 load_state_dict

        return self # 返回训练好的模型实例

    def predict(self, X):
        """
        进行二分类预测 (返回 0 或 1)。
        默认阈值是 0.5。
        """
        proba = self.predict_proba(X) # 先获取概率
        return (proba >= 0.5).astype(int) # 根据概率和 0.5 阈值得到类别

    def predict_proba(self, X):
        """
        获取正类 (类别 1) 的概率估计 (返回一维 numpy 数组)。
        """
        self.model.eval() # 切换到评估模式
        if isinstance(X, pd.DataFrame): X = X.values # 转成 numpy
        # 确保输入是二维的 [N, features]，即使 N=1
        if X.ndim == 1:
            X = X.reshape(1, -1) # 如果输入是一维的，变成 [1, features]

        X_tensor = torch.FloatTensor(X).to(self.device) # 转成 Tensor 并放到设备上
        with torch.no_grad(): # 预测时不需要梯度
            logits = self.model(X_tensor) # 模型输出 logits
            y_proba = torch.sigmoid(logits).cpu().numpy() # 应用 sigmoid 得到概率，并转回 numpy

        return y_proba.reshape(-1) # 返回一个一维数组 [N,]

def create_model(model_type, **kwargs):
    """
    根据类型创建对应的模型实例。

    参数:
        model_type (str): 模型类型，应该是 'LR', 'SVM', 或 'ANN' (不区分大小写)。
        **kwargs: 传给具体模型构造函数的参数。

    返回:
        一个初始化好的模型实例 (具有 fit, predict, predict_proba 方法)。
    """
    model_type_upper = model_type.upper() # 转成大写方便比较
    if model_type_upper == 'LR':
        return LogisticRegressionModel(**kwargs)
    elif model_type_upper == 'SVM':
        return SVMModel(**kwargs)
    elif model_type_upper == 'MLP':
        return ANNModel(**kwargs)

# --- 下面是示例用法，展示如何使用 ANNModel ---
if __name__ == '__main__':
    # 导入需要的 scikit-learn 工具
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    logger.info("开始运行模型文件 (models.py) 的示例代码...")

    # 1. 创建一些模拟的二分类数据
    # n_samples: 样本数, n_features: 总特征数, n_informative: 有用特征数
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    logger.info(f"创建了模拟数据: 特征形状={X.shape}, 标签形状={y.shape}")

    # 2. 划分数据集：训练集 (60%)、验证集 (20%)、测试集 (20%)
    # 先分出测试集 (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # stratify=y 保证各类别比例一致
    )
    # 再从剩余的 (80%) 中分出验证集 (占剩余的 25%，即总体的 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, # 0.25 * 0.8 = 0.2
        random_state=42, stratify=y_train_val
    )

    logger.info(f"数据集划分: 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")

    # 3. 特征标准化 (非常重要！特别是对于神经网络和SVM)
    scaler = StandardScaler()
    # 在训练集上学习 (fit) 并转换 (transform)
    X_train_scaled = scaler.fit_transform(X_train)
    # 用训练集学习到的 scaler 来转换验证集和测试集
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info("已对特征进行标准化处理。")

    # 4. 初始化 ANN 模型包装器
    input_dim = X_train_scaled.shape[1] # 输入维度就是特征数量
    ann_model_instance = ANNModel(
        input_dim=input_dim,
        hidden_dim1=32,      # 对小数据集用小一点的隐藏层
        hidden_dim2=16,
        dropout_rate=0.3,    # 可以调整 dropout
        learning_rate=0.005, # 学习率
        batch_size=16,       # 小数据集用小批次可能效果更好
        num_epochs=200,      # 设置一个比较大的最大轮数，让早停来决定何时结束
        weight_decay=0.001,  # L2 正则化
        early_stopping_patience=15, # 早停耐心值
        lr_scheduler_type='plateau', # 使用 Plateau 学习率调度器
        lr_scheduler_patience=7,     # Plateau 的耐心值
        random_state=42
    )
    logger.info(f"已初始化 ANN 模型: {ann_model_instance.name}")

    # 5. 训练模型 (传入训练集和验证集)
    logger.info("--- 开始训练 ANN 模型 ---")
    # 调用 fit 方法，它内部会处理训练循环、验证、早停、学习率调整
    ann_model_instance.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val)
    logger.info("--- ANN 模型训练完成 ---")

    # 6. 在测试集上进行评估
    logger.info("--- 在测试集上评估模型 ---")
    # 获取预测概率
    y_pred_proba_test = ann_model_instance.predict_proba(X_test_scaled)
    # 获取预测类别 (默认阈值 0.5)
    y_pred_test = ann_model_instance.predict(X_test_scaled)

    # 计算准确率
    accuracy_test = accuracy_score(y_test, y_pred_test)
    # 计算 AUC
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    logger.info(f"测试集 准确率 (Accuracy): {accuracy_test:.4f}")
    logger.info(f"测试集 AUC: {auc_test:.4f}")

    logger.info("模型文件 (models.py) 示例代码运行结束。")