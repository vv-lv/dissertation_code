import hashlib
import os
import glob
import pickle
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# ----- 需要你自行实现或引入以下文件（保持与原有项目路径一致） -----
from Resnet import BinaryResNet3D_ParallelBranch, BinaryResNet3D_SingleBranch
from DenseNet3D import DenseNet3DParallelBranch, DenseNet3DMRIOnlyParallelInput
from dataset_edge import MRIDataset3D

# ================ 1) 全局随机种子 ===================
def set_seed(seed=42):
    """
    保证实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"seed={seed} 设置成功")

# ================ 2) 超参数 ===================
HYPERPARAMS = {
    # 基本训练参数
    "batch_size": 6,
    "num_epochs": 100,
    "learning_rate": 5e-4,  # 基础学习率
    "weight_decay": 3e-4,   # 权重衰减
    "n_splits": 5,          # k折交叉验证
    
    # 模型参数
    "model_depth": 18,      # ResNet深度
    "num_input_channels": 1, # 每个分支的输入通道数
    "fixed_depth": 32,      # 固定图像深度
    
    # 预训练相关
    "use_pretrained": True,
    "pretrained_path": "/home/vipuser/Desktop/MedicalNet/pretrain/resnet_50_23dataset.pth",
    "freeze_layers": True,  # 是否冻结早期层
    "init_mask_from_mri": True, # 使用MRI分支权重初始化掩码分支
    
    # 学习率策略
    "use_layerwise_lr": True,   # 使用分层学习率
    "lr_factor_pretrained": 0.02, # 预训练层学习率因子
    "lr_factor_new": 1.0,      # 新层学习率因子
    
    # 解冻策略
    "progressive_unfreeze": True, # 启用解冻策略
    "use_dynamic_unfreezing": True, # 使用动态解冻替代基于epoch的解冻
    "unfreeze_monitor": "val_auc", # 监控指标 'val_acc' 或 'val_auc'
    "plateau_patience": 5,      # 性能停滞多少个epoch后解冻
    "unfreeze_min_delta": 0.003,# 性能改善最小阈值
    "min_epochs_between_unfreeze": 8, # 两次解冻之间的最小epoch数
    
    # 添加学习率预热相关配置
    "use_lr_warmup": True,    # 是否在解冻后使用学习率预热
    "warmup_epochs": 4,       # 预热持续的epoch数
    "warmup_factor": 0.05,     # 预热期间学习率降低的因子
    
    # 传统基于epoch的解冻映射(当不使用动态解冻时使用)
    "unfreeze_epoch_map":{
        # Phase 1: Begin with unfreezing the later parts of layer3
        25: ['mri_layer3.5', 'mri_layer3.4', 'mask_layer3.5', 'mask_layer3.4'],
    },
    'layer_groups':[
                        # 1. 最先解冻layer3.0，因为它是当前冻结层中最深的
                        ['layer3.0'],
                        
                        # 2. 解冻layer2的最后部分，保持更早层冻结
                        ['mri_layer2.3', 'mask_layer2.3'],
                        
                        # 3. 解冻layer2的中间部分
                        ['mri_layer2.2', 'mask_layer2.2'],
                        
                        # 4. 解冻layer2的前半部分
                        ['mri_layer2.1', 'mri_layer2.0', 'mask_layer2.1', 'mask_layer2.0'],
                        
                        # 5. 解冻layer1后半部分
                        ['mri_layer1.1', 'mask_layer1.1'],
                        
                        # 6. 解冻layer1前半部分和初始层
                        ['mri_layer1.0', 'mri_conv1', 'mri_bn1', 'mask_layer1.0', 'mask_conv1', 'mask_bn1']
                    ],
    'frozen_layers':[
            'mri_conv1', 'mri_bn1', 'mri_layer1', 'mri_layer2', 
            'mask_conv1', 'mask_bn1', 'mask_layer1', 'mask_layer2',
            # 初始不冻结layer3的大部分，让模型有更多适应空间
            'layer3.0'
        ],

    # 学习率调度器
    "scheduler_mode": "exponential",  # 可选 "plateau", "step", "cosine", "exponential"
    "step_size": 5,
    "gamma": 0.7,
    "plateau_factor": 0.5,
    "plateau_patience": 5,
    "cosine_Tmax": 100,
    "exponential_gamma": 0.985,
    
    # 早停
    "earlystop_patience": 20,
    "earlystop_min_delta": 0.003,
    
    # 其他参数
    "checkpoint_interval": 5,  # 训练多少个epoch保存一次
    "plot_interval": 1,        # 绘图间隔
    "results_dir": "../results", # 结果保存目录
    "use_balanced_loss": True,  # 使用平衡损失函数
    "pos_weight": 2.0,          # 正样本权重
    "use_mixed_precision": True, # 使用混合精度训练
    "clip_grad_norm": 1.0,      # 梯度裁剪阈值
}

HYPERPARAMS.update({
    "use_pretrained": False,     # 禁用预训练权重加载
    "freeze_layers": False,      # 禁用层冻结
    "progressive_unfreeze": False, # 禁用渐进式解冻机制
    "use_dynamic_unfreezing": False, # 禁用动态解冻策略
    "use_layerwise_lr": False,  # 所有层使用相同的学习率
    'num_epochs': 48
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================ 3) EarlyStopping类 ===================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, monitor='val_acc', mode='max'):
        """
        早停机制
        
        参数:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善阈值
            monitor: 监控指标，'val_acc'或'val_loss'
            mode: 'min'用于损失，'max'用于准确率
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        # 确保模式和监控指标匹配
        if mode == 'min' and monitor == 'val_acc':
            print("警告: 使用min模式监控val_acc可能不合适，已自动切换为max模式")
            self.mode = 'max'
        elif mode == 'max' and monitor == 'val_loss':
            print("警告: 使用max模式监控val_loss可能不合适，已自动切换为min模式")
            self.mode = 'min'

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] 连续 {self.counter}/{self.patience} 个epoch未显著提升.")
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] 连续 {self.counter}/{self.patience} 个epoch未显著提升.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ================ 4) 构建学习率调度器 ===================
def build_scheduler(optimizer, hp):
    """构建学习率调度器"""
    mode = hp["scheduler_mode"].lower()
    
    if mode == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=hp["plateau_factor"],
            patience=hp["plateau_patience"],
            verbose=True
        )
    elif mode == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp["cosine_Tmax"],
            eta_min=hp["learning_rate"] / 20  # 最小学习率为初始值的1/20
        )
    elif mode == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=hp["exponential_gamma"]
        )
    else:  # 默认step
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=hp["step_size"],
            gamma=hp["gamma"]
        )

# ================ 5) 验证/测试阶段的评估函数 ===================
def validate(val_loader, model, criterion):
    """验证/测试模型性能"""
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

    val_loss /= len(val_loader.dataset)
    
    # 计算各项指标
    correct = sum([1 for p, t in zip(all_preds, all_labels) if p == t])
    val_acc = correct / len(all_labels)

    # 计算AUC
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = 0.0

    # 计算F1分数
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    
    # 混淆矩阵
    conf_mat = confusion_matrix(all_labels, all_preds)

    return val_loss, val_acc, val_auc, val_f1, conf_mat

# ================ 6) 绘制训练过程曲线 ===================
def plot_training_progress(fold, train_losses, train_accs, val_losses, val_accs, val_aucs, val_f1s, fold_dir, unfreeze_epochs=None):
    """绘制训练指标和解冻点"""
    epochs = range(1, len(train_losses) + 1)
    
    # 创建2x2的图表布局
    plt.figure(figsize=(15, 10))
    
    # 左上: 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Training & Validation Loss (Fold {fold+1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加解冻点标记
    if unfreeze_epochs:
        for e in unfreeze_epochs:
            if e < len(train_losses):
                plt.axvline(x=e+1, color='g', linestyle='--', alpha=0.5)
    
    # 右上: 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title(f'Training & Validation Accuracy (Fold {fold+1})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加解冻点标记
    if unfreeze_epochs:
        for e in unfreeze_epochs:
            if e < len(train_accs):
                plt.axvline(x=e+1, color='g', linestyle='--', alpha=0.5)
    
    # 左下: AUC曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_aucs, 'g-')
    plt.title(f'Validation AUC (Fold {fold+1})')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加解冻点标记
    if unfreeze_epochs:
        for e in unfreeze_epochs:
            if e < len(val_aucs):
                plt.axvline(x=e+1, color='g', linestyle='--', alpha=0.5)
    
    # 右下: F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_f1s, 'purple')
    plt.title(f'Validation F1 Score (Fold {fold+1})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加解冻点标记
    if unfreeze_epochs:
        for e in unfreeze_epochs:
            if e < len(val_f1s):
                plt.axvline(x=e+1, color='g', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存到单一文件，每次更新覆盖
    plt.savefig(os.path.join(fold_dir, "training_metrics.png"))
    
    # # 同时保存带时间戳的版本
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # plt.savefig(os.path.join(fold_dir, f"training_metrics_{timestamp}.png"))
    
    plt.close()

# ================ 7) 模型初始化与层冻结 ===================
def get_model(hp):
    """创建并初始化带有扩展冻结的模型"""
    # model = BinaryResNet3D_ParallelBranch(
    #     model_depth=hp['model_depth'],
    #     num_input_channels=hp['num_input_channels']
    # )
    # model = BinaryResNet3D_SingleBranch(
    #     model_depth=hp['model_depth'],
    #     num_input_channels=hp['num_input_channels']
    # )
    # model = BinaryResNet3D_ParallelBranch(
    #     model_depth=34,
    #     num_input_channels=hp['num_input_channels']
    # )
    # model = BinaryResNet3D_SingleBranch(
    #     model_depth=34,
    #     num_input_channels=hp['num_input_channels']
    # )
    # model = BinaryResNet3D_ParallelBranch(
    #     model_depth=50,
    #     num_input_channels=hp['num_input_channels']
    # )
    model = BinaryResNet3D_SingleBranch(
        model_depth=50,
        num_input_channels=hp['num_input_channels']
    )
    # 加载预训练权重
    if hp['use_pretrained'] and hp['pretrained_path']:
        print(f"从 {hp['pretrained_path']} 加载预训练权重...")
        model.load_pretrained_weights(
            pretrained_path=hp['pretrained_path'],
            init_mask_from_mri=hp['init_mask_from_mri']
        )
    
    # 更广泛的冻结 - 包括layer2和部分layer3
    if hp['freeze_layers']:
        frozen_count = 0
        print("冻结更广泛的特征提取层...")
        # 冻结策略修改
        frozen_layers = hp['frozen_layers']
        
        for name, param in model.named_parameters():
            if any(x in name for x in frozen_layers):
                param.requires_grad = False
                frozen_count += 1  # 这行是修正，需要增加计数器
                
        print(f"已冻结 {frozen_count} 个参数组")
    
    return model

# ================ 8) 渐进式解冻调度器 ===================
class ProgressiveUnfreezing:
    def __init__(self, model, unfreeze_epoch_map=None):
        """
        基于预设epoch的渐进式解冻调度器
        
        参数:
            model: 要解冻的模型
            unfreeze_epoch_map: 解冻映射，格式 {epoch: [layer_patterns]}
        """
        self.model = model
        self.unfreeze_epoch_map = unfreeze_epoch_map or {}
    
    def step(self, epoch):
        """执行解冻步骤"""
        if epoch in self.unfreeze_epoch_map:
            layer_patterns = self.unfreeze_epoch_map[epoch]
            unfrozen_count = 0
            unfrozen_layers = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad and any(pattern in name for pattern in layer_patterns):
                    param.requires_grad = True
                    unfrozen_count += 1
                    unfrozen_layers.append(name)
            
            if unfrozen_count > 0:
                print(f"\n[渐进式解冻] 在epoch {epoch+1}解冻了{unfrozen_count}个参数组")
                print(f"解冻的层: {layer_patterns}")
                print(f"解冻的参数: {unfrozen_layers}")
                
                # 计算仍然冻结的参数数量
                still_frozen = sum(1 for name, param in self.model.named_parameters() if not param.requires_grad)
                total_params = sum(1 for name, param in self.model.named_parameters())
                print(f"当前状态: {still_frozen}/{total_params} 参数仍然冻结")
            
            return True
        return False

# ================ 9) 动态解冻调度器 ===================
class DynamicUnfreezing:
    def __init__(self, model, layer_groups, monitor='val_acc', plateau_patience=3, min_delta=0.001, min_epochs_between_unfreeze=3):
        """
        基于性能停滞动态解冻
        
        参数:
            model: 要解冻的模型
            layer_groups: 解冻层的分组列表，按解冻顺序排列
            monitor: 监控的指标，'val_acc'或'val_auc'
            plateau_patience: 性能停滞多少个epoch后触发解冻
            min_delta: 性能提升的最小阈值
            min_epochs_between_unfreeze: 两次解冻之间的最小epoch数
        """
        self.model = model
        self.layer_groups = layer_groups
        self.monitor = monitor
        self.plateau_patience = plateau_patience
        self.min_delta = min_delta
        self.current_group = 0
        self.best_score = None
        self.plateau_counter = 0
        self.last_unfrozen_epoch = -1  # 记录上次解冻的epoch
        self.min_epochs_between_unfreeze = min_epochs_between_unfreeze
        self.unfrozen_epochs = []  # 记录解冻发生的epoch
        
    def step(self, current_score, epoch):
        """基于当前性能评估是否需要解冻下一组层，并返回是否需要预热"""
        # 如果所有层组都已解冻，直接返回
        if self.current_group >= len(self.layer_groups):
            return False, False  # 第二个返回值表示不需要预热
            
        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = current_score
            return False, False
        
        # 距离上次解冻的epoch太近，不执行解冻
        if epoch - self.last_unfrozen_epoch < self.min_epochs_between_unfreeze:
            # 仍然更新最佳分数
            if (self.monitor == 'val_acc' and current_score > self.best_score + self.min_delta) or \
            (self.monitor == 'val_auc' and current_score > self.best_score + self.min_delta):
                self.best_score = current_score
                self.plateau_counter = 0
            else:
                self.plateau_counter += 1
            return False, False
            
        # 检查性能是否有显著提升
        if (self.monitor == 'val_acc' and current_score > self.best_score + self.min_delta) or \
        (self.monitor == 'val_auc' and current_score > self.best_score + self.min_delta):
            self.best_score = current_score
            self.plateau_counter = 0
            return False, False
        else:
            self.plateau_counter += 1
            
        # 如果性能停滞达到耐心阈值，解冻下一组层
        if self.plateau_counter >= self.plateau_patience and self.current_group < len(self.layer_groups):
            layer_patterns = self.layer_groups[self.current_group]
            unfrozen_count = 0
            unfrozen_layers = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad and any(pattern in name for pattern in layer_patterns):
                    param.requires_grad = True
                    unfrozen_count += 1
                    unfrozen_layers.append(name)
            
            if unfrozen_count > 0:
                print(f"\n[动态解冻] 在epoch {epoch+1}，发现性能停滞{self.plateau_counter}个epoch")
                print(f"解冻第{self.current_group+1}/{len(self.layer_groups)}组层: {layer_patterns}")
                print(f"解冻的参数: {unfrozen_layers}")
                
                # 计算仍然冻结的参数数量
                still_frozen = sum(1 for name, param in self.model.named_parameters() if not param.requires_grad)
                total_params = sum(1 for name, param in self.model.named_parameters())
                print(f"当前状态: {still_frozen}/{total_params} 参数仍然冻结")
                
                self.current_group += 1
                self.plateau_counter = 0
                self.last_unfrozen_epoch = epoch
                self.unfrozen_epochs.append(epoch)
                return True, True  # 第二个返回值表示需要预热
            else:
                # 如果当前组没有可解冻的层，移动到下一组
                self.current_group += 1
                return self.step(current_score, epoch)  # 递归尝试下一组
                
        return False, False

    def get_unfrozen_epochs(self):
        """返回解冻发生的epoch列表"""
        return self.unfrozen_epochs

# ================ 10) 分层学习率优化器 ===================
def get_optimizer_with_layerwise_lr(model, hp):
    """
    为不同层设置不同的学习率和权重衰减
    
    参数:
        model: 要优化的模型
        hp: 超参数字典
        
    返回:
        配置好的优化器
    """
    if not hp['use_layerwise_lr']:
        return optim.AdamW(
            model.parameters(), 
            lr=hp['learning_rate'], 
            weight_decay=hp['weight_decay']
        )
    
    # 将参数分为三组：
    # 1. 冻结层 (lr=0)
    # 2. 预训练层但未冻结 (lr=base_lr*lr_factor_pretrained)
    # 3. 新初始化层 (lr=base_lr*lr_factor_new)
    frozen_params = []
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(param)
        elif any(x in name for x in ['mri_', 'mask_', 'layer3', 'layer4']):
            # 预训练层使用较小的学习率
            pretrained_params.append(param)
        else:
            # 分类器等新层使用较大的学习率
            new_params.append(param)
    
    # 打印各组参数数量
    print(f"分层学习率 - 冻结参数: {len(frozen_params)}, 预训练参数: {len(pretrained_params)}, 新参数: {len(new_params)}")
    
    # 使用AdamW优化器并设置不同的学习率和权重衰减
    optimizer = optim.AdamW([
        {'params': frozen_params, 'lr': 0},
        {'params': pretrained_params, 
         'lr': hp['learning_rate'] * hp['lr_factor_pretrained'], 
         'weight_decay': hp['weight_decay']},
        {'params': new_params, 
         'lr': hp['learning_rate'] * hp['lr_factor_new'], 
         'weight_decay': hp['weight_decay'] * 5}  # 新层使用更大的权重衰减
    ])
    
    return optimizer

# ================ 11) 单折训练函数 ===================
def train_one_fold(
    fold, 
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    optimizer, 
    scheduler, 
    hp,
    fold_dir,
    unfreeze_scheduler=None
):
    """训练一折的函数"""
    best_val_acc = 0.0
    best_val_auc = 0.0
    early_stopper = EarlyStopping(
        patience=hp["earlystop_patience"], 
        min_delta=hp["earlystop_min_delta"],
        monitor='val_acc',
        mode='max'
    )

    # 添加学习率预热跟踪变量
    warmup_active = False
    warmup_counter = 0
    warmup_epochs = hp.get("warmup_epochs", 3)  # 预热持续的epoch数
    warmup_factor = hp.get("warmup_factor", 0.1)  # 预热期间学习率降低的因子
    original_lr_factors = None  # 存储原始学习率因子
    
    # 保存日志
    log_path = os.path.join(fold_dir, "training_log.txt")
    unfreezing_log_path = os.path.join(fold_dir, "unfreezing_log.txt")
    
    # 记录训练指标
    train_losses, train_accs = [], []
    val_losses, val_accs, val_aucs, val_f1s = [], [], [], []
    
    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if hp['use_mixed_precision'] and torch.cuda.is_available() else None
    
    start_time = time.time()
    with open(log_path, 'w') as f_log, open(unfreezing_log_path, 'w') as f_unfreeze:
        f_unfreeze.write("Epoch,Unfrozen_Layer_Group,Still_Frozen_Params,Total_Params,Warmup\n")
        
        header = "Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Val AUC,Val F1,LR\n"
        f_log.write(header)
        
        for epoch in range(hp["num_epochs"]):
            # 处理预热恢复
            if warmup_active:
                warmup_counter += 1
                # 计算当前预热进度
                progress = min(warmup_counter / warmup_epochs, 1.0)
                
                # 如果使用分层学习率
                if hp['use_layerwise_lr'] and len(optimizer.param_groups) > 1:
                    # 逐步恢复学习率
                    for i, factor in enumerate(original_lr_factors):
                        if i > 0:  # 跳过冻结参数组
                            current_lr = hp['learning_rate'] * factor * (warmup_factor + (1.0 - warmup_factor) * progress)
                            optimizer.param_groups[i]['lr'] = current_lr
                
                # 完成预热
                if warmup_counter >= warmup_epochs:
                    warmup_active = False
                    # 恢复到正常的分层学习率
                    if hp['use_layerwise_lr'] and len(optimizer.param_groups) > 1 and original_lr_factors:
                        for i, factor in enumerate(original_lr_factors):
                            if i > 0:  # 跳过冻结参数组
                                optimizer.param_groups[i]['lr'] = hp['learning_rate'] * factor
                    
                    print(f"[学习率预热] 在epoch {epoch+1}完成预热，恢复正常学习率")
            
            # 检查解冻
            unfreeze_occurred = False
            need_warmup = False
            
            if unfreeze_scheduler:
                if isinstance(unfreeze_scheduler, DynamicUnfreezing):
                    # 动态解冻基于性能
                    monitor_value = val_accs[-1] if val_accs and unfreeze_scheduler.monitor == 'val_acc' else val_aucs[-1] if val_aucs else 0
                    if epoch > 0:
                        unfreeze_occurred, need_warmup = unfreeze_scheduler.step(monitor_value, epoch)
                        
                        # 如果解冻发生并需要预热
                        if unfreeze_occurred and need_warmup and hp.get("use_lr_warmup", True):
                            # 重新配置优化器
                            optimizer = get_optimizer_with_layerwise_lr(model, hp)
                            
                            # 保存原始学习率因子
                            original_lr_factors = []
                            for group in optimizer.param_groups:
                                factor = group['lr'] / hp['learning_rate'] if hp['learning_rate'] > 0 else 0
                                original_lr_factors.append(factor)
                            
                            # 设置预热降低的学习率（除了冻结参数组）
                            for i in range(1, len(optimizer.param_groups)):
                                optimizer.param_groups[i]['lr'] *= warmup_factor
                            
                            # 启动预热
                            warmup_active = True
                            warmup_counter = 0
                            
                            print(f"[学习率预热] 在epoch {epoch+1}开始预热，学习率降低到原来的{warmup_factor*100}%")
                            
                            # 记录解冻和预热信息
                            still_frozen = sum(1 for name, param in model.named_parameters() if not param.requires_grad)
                            total_params = sum(1 for name, param in model.named_parameters())
                            f_unfreeze.write(f"{epoch+1},{unfreeze_scheduler.current_group-1},{still_frozen},{total_params},warmup_start\n")
                            f_unfreeze.flush()
                        elif unfreeze_occurred:
                            # 如果只是解冻但不需要预热，仍然重新配置优化器
                            optimizer = get_optimizer_with_layerwise_lr(model, hp)
                            
                            # 记录解冻信息
                            still_frozen = sum(1 for name, param in model.named_parameters() if not param.requires_grad)
                            total_params = sum(1 for name, param in model.named_parameters())
                            f_unfreeze.write(f"{epoch+1},{unfreeze_scheduler.current_group-1},{still_frozen},{total_params},no_warmup\n")
                            f_unfreeze.flush()
                else:
                    # 基于epoch的解冻，保持不变
                    if unfreeze_scheduler.step(epoch):
                        optimizer = get_optimizer_with_layerwise_lr(model, hp)
                        unfreeze_occurred = True
                        
                        still_frozen = sum(1 for name, param in model.named_parameters() if not param.requires_grad)
                        total_params = sum(1 for name, param in model.named_parameters())
                        f_unfreeze.write(f"{epoch+1},epoch_based,{still_frozen},{total_params},no_warmup\n")
                        f_unfreeze.flush()
            
            # 训练一个epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            epoch_start_time = time.time()
            
            # 训练循环部分保持不变...
            for images, labels in tqdm(train_loader, desc=f"[Fold {fold+1}] Epoch {epoch+1}/{hp['num_epochs']}"):
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    # 缩放损失并反向传播
                    scaler.scale(loss).backward()
                    
                    # 梯度裁剪以防止梯度爆炸
                    if hp.get('clip_grad_norm', 0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['clip_grad_norm'])
                    
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 常规训练流程
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    if hp.get('clip_grad_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['clip_grad_norm'])
                        
                    optimizer.step()
                
                running_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # 验证阶段
            val_loss, val_acc, val_auc, val_f1, conf_mat = validate(val_loader, model, criterion)
            
            # 学习率调度
            if hp["scheduler_mode"] == "plateau":
                scheduler.step(val_loss)  # 使用验证损失
            else:
                scheduler.step()
            
            # 获取当前学习率
            if isinstance(optimizer, optim.AdamW) and len(optimizer.param_groups) > 1:
                current_lr = optimizer.param_groups[2]['lr']  # 使用新参数组的学习率
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # 记录指标
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_aucs.append(val_auc)
            val_f1s.append(val_f1)
            
            # 计算epoch用时
            epoch_time = time.time() - epoch_start_time
            
            # 创建日志字符串
            log_str = (
                f"Epoch {epoch+1}/{hp['num_epochs']} [{epoch_time:.1f}s] "
                f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% "
                f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% "
                f"| Val AUC: {val_auc:.3f}, Val F1: {val_f1:.3f} "
                f"| LR: {current_lr:.6f}"
            )
            
            if unfreeze_occurred:
                log_str += " | 本轮解冻✓"
            if warmup_active:
                log_str += f" | 预热中({warmup_counter}/{warmup_epochs})"
                
            print(log_str)
            
            # 写入CSV格式日志
            csv_log = f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_auc:.6f},{val_f1:.6f},{current_lr:.6f}\n"
            f_log.write(csv_log)
            f_log.flush()
            
            # 保存检查点
            if (epoch + 1) % hp["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(fold_dir, "checkpoint_latest.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'conf_mat': conf_mat.tolist(),
                }, ckpt_path)
                print(f"   => [Checkpoint] Epoch {epoch+1} -> 'checkpoint_latest.pth'")
            
            # 保存最佳模型 (基于验证准确率)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(fold_dir, "best_model_acc.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'conf_mat': conf_mat.tolist(),
                }, best_model_path)
                print(f"   => [最佳准确率] 新记录: {val_acc*100:.2f}% (之前: {best_val_acc*100:.2f}%)")
            
            # 保存最佳AUC模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_auc_path = os.path.join(fold_dir, "best_model_auc.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'conf_mat': conf_mat.tolist(),
                }, best_auc_path)
                print(f"   => [最佳AUC] 新记录: {val_auc:.4f} (之前: {best_val_auc:.4f})")
            
            # 绘制训练曲线
            if (epoch + 1) % hp["plot_interval"] == 0:
                unfreeze_epochs = unfreeze_scheduler.get_unfrozen_epochs() if isinstance(unfreeze_scheduler, DynamicUnfreezing) else None
                plot_training_progress(
                    fold, train_losses, train_accs, 
                    val_losses, val_accs, val_aucs, val_f1s, 
                    fold_dir, unfreeze_epochs
                )
            
            # 检查早停
            early_stopper(val_acc)
            if early_stopper.early_stop:
                stop_str = f"[EarlyStopping] 在epoch {epoch+1}触发提前停止.\n"
                print(stop_str)
                f_log.write(stop_str)
                break
    
    # 记录解冻的最终状态
    with open(unfreezing_log_path, 'a') as f_unfreeze:
        still_frozen = sum(1 for name, param in model.named_parameters() if not param.requires_grad)
        total_params = sum(1 for name, param in model.named_parameters())
        f_unfreeze.write(f"Final,{still_frozen},{total_params}\n")
    
    # 训练结束，记录总结
    total_time = time.time() - start_time
    end_str = (
        f"Fold {fold+1}/{hp['n_splits']}完成. "
        f"最佳Val Acc: {best_val_acc*100:.2f}%, "
        f"最佳Val AUC: {best_val_auc:.4f}, "
        f"总用时: {total_time:.2f}s\n"
    )
    print(end_str)
    with open(log_path, 'a') as f_log:
        f_log.write("\n" + end_str)
    
    # 保存最终模型
    final_path = os.path.join(fold_dir, "final_model.pth")
    torch.save({
        'epoch': hp['num_epochs'],
        'model_state_dict': model.state_dict(),
        'val_acc': val_accs[-1] if val_accs else 0,
        'val_auc': val_aucs[-1] if val_aucs else 0,
        'val_f1': val_f1s[-1] if val_f1s else 0,
    }, final_path)
    print(f"   => 最终模型已保存 (Fold {fold+1}).")
    
    # 绘制最终训练曲线
    unfreeze_epochs = unfreeze_scheduler.get_unfrozen_epochs() if isinstance(unfreeze_scheduler, DynamicUnfreezing) else None
    plot_training_progress(
        fold, train_losses, train_accs, 
        val_losses, val_accs, val_aucs, val_f1s, 
        fold_dir, unfreeze_epochs
    )
    
    return best_val_acc, best_val_auc

# ================ 12) 交叉验证主函数 ===================
def cross_validate(train_dataset, hp, test_dataset=None, only_test_dataset=None):
    """
    执行k折交叉验证并可选地在测试集上评估集成模型
    
    参数:
        train_dataset: 训练数据集
        hp: 超参数字典
        test_dataset: 可选的测试数据集
    
    返回:
        折平均指标
    """
    if not only_test_dataset:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 准备结果目录
        results_dir = hp["results_dir"]
        experiment_name = f"resnet{hp['model_depth']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = os.path.join(results_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存超参数
        with open(os.path.join(exp_dir, "hyperparameters.txt"), 'w') as f:
            for key, value in hp.items():
                f.write(f"{key}: {value}\n")
        
        # 初始化k折交叉验证
        kf = KFold(n_splits=hp["n_splits"], shuffle=True, random_state=42)
        
        # 用于记录每折结果
        fold_results = []
        
        # 开始k折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            print(f"\n{'='*20} Fold {fold+1}/{hp['n_splits']} {'='*20}")
            
            # 为当前折创建目录
            fold_dir = os.path.join(exp_dir, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # 构建子集
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_subset, 
                batch_size=hp["batch_size"], 
                shuffle=True,
                num_workers=4,   # 根据您的CPU核心数调整
                pin_memory=True  # 加速数据传输
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=hp["batch_size"], 
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # 创建模型
            model = get_model(hp).to(device)
            
            # 创建损失函数
            if hp["use_balanced_loss"]:
                # 使用加权BCE损失以处理类别不平衡
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([hp["pos_weight"]]).to(device)
                )
            else:
                criterion = nn.BCEWithLogitsLoss()
            
            # 创建优化器（使用分层学习率）
            optimizer = get_optimizer_with_layerwise_lr(model, hp)
            
            # 创建学习率调度器
            scheduler = build_scheduler(optimizer, hp)
            
            # 创建解冻调度器
            unfreeze_scheduler = None
            if hp["progressive_unfreeze"] and hp["freeze_layers"]:
                if hp.get("use_dynamic_unfreezing", False):
                    # 定义解冻层的分组（从上到下）
                    layer_groups = hp['layer_groups']
                    
                    unfreeze_scheduler = DynamicUnfreezing(
                        model=model,
                        layer_groups=layer_groups,
                        monitor=hp.get("unfreeze_monitor", "val_acc"),
                        plateau_patience=hp.get("plateau_patience", 3),
                        min_delta=hp.get("unfreeze_min_delta", 0.002),
                        min_epochs_between_unfreeze=hp.get("min_epochs_between_unfreeze", 3)
                    )
                    print(f"使用动态解冻策略，监控 {hp.get('unfreeze_monitor', 'val_acc')}，耐心值 {hp.get('plateau_patience', 3)}")
                else:
                    # 基于epoch的解冻
                    unfreeze_scheduler = ProgressiveUnfreezing(
                        model, 
                        unfreeze_epoch_map=hp["unfreeze_epoch_map"]
                    )
                    print(f"使用基于epoch的渐进式解冻策略")
            
            # 训练当前折
            best_acc, best_auc = train_one_fold(
                fold=fold,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                hp=hp,
                fold_dir=fold_dir,
                unfreeze_scheduler=unfreeze_scheduler
            )
            
            # 记录此折结果
            fold_results.append({
                'fold': fold + 1,
                'best_acc': best_acc,
                'best_auc': best_auc
            })
        
        # 打印交叉验证总结
        print("\n" + "="*50)
        print(f"交叉验证完成 - {experiment_name}")
        print("="*50)
        
        # 计算平均指标
        avg_acc = np.mean([r['best_acc'] for r in fold_results])
        avg_auc = np.mean([r['best_auc'] for r in fold_results])
        
        print(f"平均验证集准确率: {avg_acc*100:.2f}%")
        print(f"平均验证集AUC: {avg_auc:.4f}")
        print("\n各折结果:")
        
        for r in fold_results:
            print(f"Fold {r['fold']}: Acc={r['best_acc']*100:.2f}%, AUC={r['best_auc']:.4f}")
        
        # 保存交叉验证结果摘要
        summary_path = os.path.join(exp_dir, "cv_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"实验: {experiment_name}\n")
            f.write(f"平均验证集准确率: {avg_acc*100:.2f}%\n")
            f.write(f"平均验证集AUC: {avg_auc:.4f}\n\n")
            f.write("各折结果:\n")
            for r in fold_results:
                f.write(f"Fold {r['fold']}: Acc={r['best_acc']*100:.2f}%, AUC={r['best_auc']:.4f}\n")
    else:
        exp_dir = only_test_dataset
        
    if test_dataset is not None:
        print("\n" + "="*50)
        print(f"在测试集上评估交叉验证模型集成...")
        print("="*50)
        
        # 评估集成模型
        test_acc, test_auc, test_f1, conf_mat = ensemble_cv_models_on_test(exp_dir, test_dataset, hp)
        
        # 保存测试结果到摘要文件
        test_summary_path = os.path.join(exp_dir, "test_ensemble_summary.txt")
        with open(test_summary_path, 'w') as f:
            f.write(f"集成模型测试结果:\n")
            f.write(f"测试集准确率: {test_acc*100:.2f}%\n")
            f.write(f"测试集AUC: {test_auc:.4f}\n")
            f.write(f"测试集F1: {test_f1:.4f}\n")
            f.write(f"混淆矩阵:\n{conf_mat}\n")
            
            # 计算和保存更多指标
            if conf_mat is not None and conf_mat.size >= 4:
                tn, fp, fn, tp = conf_mat.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                f.write(f"\n详细指标:\n")
                f.write(f"灵敏度(召回率): {sensitivity:.4f}\n")
                f.write(f"特异度: {specificity:.4f}\n")
                f.write(f"阳性预测值(精确率): {ppv:.4f}\n")
                f.write(f"阴性预测值: {npv:.4f}\n")
                
        # 基于交叉验证结果进行多次训练和权重平均
        final_weights_path = multi_train_with_ensemble(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            hp=HYPERPARAMS,
            exp_dir=exp_dir,  # 交叉验证结果目录
            num_runs=5  # 训练次数，可以根据需要调整
        )

        print(f"\n训练完成！最终平均权重模型保存在: {final_weights_path}")
        
    # return avg_acc, avg_auc, fold_results

# ================ 12.5) 集成交叉验证模型并在测试集上评估 ===================

def determine_optimal_epochs(exp_dir, hp):
    """
    从交叉验证结果中计算最优训练轮数
    
    参数:
        exp_dir: 交叉验证结果存储目录
        hp: 超参数字典
        
    返回:
        int: 建议的最优训练轮数
    """
    best_epochs = []
    
    for fold in range(hp["n_splits"]):
        # 加载每折的最佳检查点
        best_model_path = os.path.join(exp_dir, f"fold_{fold+1}/best_model_auc.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
            best_epochs.append(checkpoint['epoch'])
        else:
            print(f"警告: 无法找到第{fold+1}折的最佳模型 {best_model_path}")
    
    if not best_epochs:
        print("警告: 未找到最佳轮数信息。使用默认值40轮。")
        return 40
    
    # 计算统计信息
    mean_epochs = np.mean(best_epochs)
    std_epochs = np.std(best_epochs)
    # 使用均值+1.5倍标准差作为建议轮数
    optimal_epochs = int(mean_epochs + 1.5 * std_epochs)
    
    print(f"各折最佳轮数: {best_epochs}")
    print(f"平均轮数: {mean_epochs:.1f}, 标准差: {std_epochs:.1f}")
    print(f"建议的最优训练轮数: {optimal_epochs}")
    
    return optimal_epochs


def train_single_model(train_dataset, test_dataset, hp, output_dir, seed, optimal_epochs):
    """
    训练单个模型
    
    参数:
        train_dataset: 完整训练数据集
        test_dataset: 测试数据集
        hp: 超参数字典
        output_dir: 输出目录
        seed: 随机种子
        optimal_epochs: 最优训练轮数
        
    返回:
        tuple: (模型路径, 验证AUC, 测试AUC)
    """
    # 设置随机种子
    set_seed(seed)
    print(f"\n使用随机种子 {seed} 训练模型...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型目录
    model_dir = os.path.join(output_dir, f"model_seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 更新超参数
    training_hp = hp.copy()
    training_hp["num_epochs"] = optimal_epochs
    
    # 保存本次训练的超参数和种子
    with open(os.path.join(model_dir, "train_info.txt"), 'w') as f:
        f.write(f"随机种子: {seed}\n")
        f.write(f"最优训练轮数: {optimal_epochs}\n\n")
        for key, value in training_hp.items():
            f.write(f"{key}: {value}\n")
    
    # 将训练集划分为训练部分和验证部分 (90%/10%)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split_idx = int(np.floor(0.1 * dataset_size))
    val_indices, train_indices = indices[:split_idx], indices[split_idx:]
    
    # 创建子集
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    print(f"训练子集大小: {len(train_subset)}")
    print(f"验证子集大小: {len(val_subset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset, 
        batch_size=training_hp["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=training_hp["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_hp["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = get_model(training_hp).to(device)
    
    # 创建损失函数
    if training_hp["use_balanced_loss"]:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([training_hp["pos_weight"]]).to(device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 创建优化器
    optimizer = get_optimizer_with_layerwise_lr(model, training_hp)
    
    # 创建学习率调度器
    scheduler = build_scheduler(optimizer, training_hp)
    
    # 设置早停
    early_stopper = EarlyStopping(
        patience=training_hp.get("earlystop_patience", 15),
        min_delta=training_hp.get("earlystop_min_delta", 0.001),
        monitor='val_auc',
        mode='max'
    )
    
    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if training_hp['use_mixed_precision'] and torch.cuda.is_available() else None
    
    # 跟踪指标
    train_losses, train_accs = [], []
    val_losses, val_accs, val_aucs, val_f1s = [], [], [], []
    
    # 最佳模型的路径
    best_model_path = os.path.join(model_dir, "best_model.pth")
    final_model_path = os.path.join(model_dir, "final_model.pth")
    
    # 记录日志
    log_path = os.path.join(model_dir, "training_log.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Val AUC,Val F1,LR\n")
    
    best_val_auc = 0.0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(training_hp["num_epochs"]):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start_time = time.time()
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_hp['num_epochs']}"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                if training_hp.get('clip_grad_norm', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_hp['clip_grad_norm'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if training_hp.get('clip_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_hp['clip_grad_norm'])
                    
                optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # 验证阶段
        val_loss, val_acc, val_auc, val_f1, conf_mat = validate(val_loader, model, criterion)
        
        # 调度器步进
        if training_hp["scheduler_mode"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 获取当前学习率
        if isinstance(optimizer, optim.AdamW) and len(optimizer.param_groups) > 1:
            current_lr = optimizer.param_groups[2]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_f1s.append(val_f1)
        
        # 计算epoch用时
        epoch_time = time.time() - epoch_start_time
        
        # 打印进度
        log_str = (
            f"Epoch {epoch+1}/{training_hp['num_epochs']} [{epoch_time:.1f}s] "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% "
            f"| Val AUC: {val_auc:.3f}, Val F1: {val_f1:.3f} "
            f"| LR: {current_lr:.6f}"
        )
        print(log_str)
        
        # 写入日志文件
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{val_auc:.6f},{val_f1:.6f},{current_lr:.6f}\n")
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'conf_mat': conf_mat.tolist(),
            }, best_model_path)
            # 同时保存仅包含权重的文件（便于权重平均）
            torch.save(model.state_dict(), os.path.join(model_dir, "best_weights.pth"))
            print(f"   => [最佳AUC] 新纪录: {val_auc:.4f} (之前: {best_val_auc:.4f})")
        
        # 检查早停
        early_stopper(val_auc)
        if early_stopper.early_stop:
            print(f"[早停] 在epoch {epoch+1}触发早停.")
            break
        
        # 绘制训练曲线
        if (epoch + 1) % training_hp.get("plot_interval", 1) == 0:
            plot_training_progress(
                0, train_losses, train_accs, 
                val_losses, val_accs, val_aucs, val_f1s, 
                model_dir, None
            )
    
    # 保存最终模型（不论是否是最佳）
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_acc': val_accs[-1],
        'val_auc': val_aucs[-1],
    }, final_model_path)
    
    # 最终训练统计
    total_time = time.time() - start_time
    print(f"训练完成。最佳验证AUC: {best_val_auc:.4f}，出现在epoch {best_epoch+1}。")
    print(f"总训练时间: {total_time:.2f}秒")
    
    # 保存最终训练曲线
    plot_training_progress(
        0, train_losses, train_accs, 
        val_losses, val_accs, val_aucs, val_f1s, 
        model_dir, None
    )
    
    # 加载最佳模型进行测试集评估
    print("\n在测试集上评估最佳模型...")
    best_model = get_model(training_hp).to(device)
    best_model.load_state_dict(torch.load(os.path.join(model_dir, "best_weights.pth")))
    test_loss, test_acc, test_auc, test_f1, test_conf_mat = validate(test_loader, best_model, criterion)
    
    print(f"测试集结果:")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"混淆矩阵:\n{test_conf_mat}")
    
    # # 保存测试结果
    # test_results_path = os.path.join(model_dir, "test_results.txt")
    # with open(test_results_path, 'w') as f:
    #     f.write("测试集结果\n")
    #     f.write("==========\n\n")
    #     f.write(f"Accuracy: {test_acc*100:.2f}%\n")
    #     f.write(f"AUC: {test_auc:.4f}\n")
    #     f.write(f"F1 Score: {test_f1:.4f}\n")
    #     f.write(f"混淆矩阵:\n{test_conf_mat}\n\n")
    
    return os.path.join(model_dir, "best_weights.pth"), best_val_auc, test_auc


def calculate_auc_with_ci(y_true, y_prob, n_bootstraps=1000, ci=95):
    """
    计算AUC及其置信区间
    
    参数:
        y_true: 真实标签
        y_prob: 预测概率
        n_bootstraps: 自助法重采样次数
        ci: 置信区间百分比（如95表示95%置信区间）
    
    返回:
        tuple: (AUC, 下限, 上限)
    """
    # 计算原始数据的AUC
    auc = roc_auc_score(y_true, y_prob)
    
    # 初始化存储自助法结果的数组
    bootstrapped_aucs = []
    
    # 如果样本太少，增加自助法采样次数
    n_bootstraps = max(n_bootstraps, 2000) if len(y_true) < 100 else n_bootstraps
    
    # 执行自助法重采样
    for _ in range(n_bootstraps):
        # 随机采样（有放回）
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        sample_true = np.array(y_true)[indices]
        sample_prob = np.array(y_prob)[indices]
        
        # 有时自助法样本中可能只有一个类别，需要处理这种情况
        if len(np.unique(sample_true)) < 2:
            continue
        
        # 计算这次自助法样本的AUC并存储
        bootstrapped_aucs.append(roc_auc_score(sample_true, sample_prob))
    
    # 计算置信区间
    alpha = (100 - ci) / 2 / 100
    lower_bound = max(0, np.percentile(bootstrapped_aucs, alpha * 100))
    upper_bound = min(1, np.percentile(bootstrapped_aucs, (1 - alpha) * 100))
    
    return auc, lower_bound, upper_bound


def ensemble_predictions(models, test_loader, device, optimal_threshold):
    """
    使用多个模型进行投票集成预测
    
    参数:
        models: 模型列表
        test_loader: 测试数据加载器
        device: 计算设备
        
    返回:
        tuple: (最终预测, 预测概率, 真实标签)
    """
    print("执行模型集成预测...")
    all_predictions = []
    all_probs = []
    true_labels = None
    
    for i, model in enumerate(models):
        # 确保模型在正确的设备上
        model = model.to(device)
        model.eval()
        
        model_preds = []
        model_probs = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"预测 - 模型 {i+1}/{len(models)}"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > optimal_threshold).astype(float)
                
                model_preds.extend(preds)
                model_probs.extend(probs)
                
                if true_labels is None:
                    labels_list.extend(labels.numpy().flatten())
            
        all_predictions.append(model_preds)
        all_probs.append(model_probs)
        
        if true_labels is None:
            true_labels = labels_list
    
    # 转换为numpy数组以便于操作
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # 多数投票
    ensemble_preds = np.round(np.mean(all_predictions, axis=0)).astype(float)
    
    # 平均概率
    ensemble_probs = np.mean(all_probs, axis=0)
    
    return ensemble_preds, ensemble_probs, true_labels


def calculate_metrics(predictions, labels, probabilities=None):
    """
    计算并返回完整的评估指标
    
    参数:
        predictions: 预测标签 (0/1)
        labels: 真实标签 (0/1)
        probabilities: 预测概率 (0-1)
        
    返回:
        dict: 包含所有评估指标的字典
    """
    # 计算混淆矩阵
    conf_mat = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = conf_mat.ravel()
    
    # 基本指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # 敏感度、特异度
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 精确率、阴性预测值
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # F1分数
    f1 = f1_score(labels, predictions, average='binary')
    
    # # AUC (需要概率值)
    # auc = roc_auc_score(labels, probabilities) if probabilities is not None else 0
    # 添加带置信区间的AUC计算
    auc, auc_lower, auc_upper = calculate_auc_with_ci(labels, probabilities)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'auc_ci_lower': auc_lower,
        'auc_ci_upper': auc_upper,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'conf_mat': conf_mat
    }


def save_metrics_report(metrics, output_path, title="模型评估指标"):
    """
    将评估指标保存为文本文件
    
    参数:
        metrics: 指标字典
        output_path: 输出文件路径
        title: 报告标题
    """
    with open(output_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("="*len(title) + "\n\n")
        
        f.write(f"准确率: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"F1分数: {metrics['f1']:.4f}\n")
        f.write(f"敏感度(召回率): {metrics['sensitivity']:.4f}\n")
        f.write(f"特异度: {metrics['specificity']:.4f}\n")
        f.write(f"精确率: {metrics['precision']:.4f}\n")
        f.write(f"阴性预测值: {metrics['npv']:.4f}\n\n")
        
        f.write(f"混淆矩阵:\n{metrics['conf_mat']}\n")


def plot_roc_curve(true_labels, probabilities, output_path, auc_value, title="ROC曲线"):
    """
    绘制ROC曲线并保存
    
    参数:
        true_labels: 真实标签
        probabilities: 预测概率
        output_path: 输出文件路径
        auc_value: AUC值
        title: 图表标题
    """
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('假阳性率 (1 - 特异度)', fontsize=12)
    plt.ylabel('真阳性率 (敏感度)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.close()


def multi_train_with_ensemble(train_dataset, test_dataset, hp, exp_dir, num_runs=3, 
                              direct_test=True, model_dir=None):
    """
    多次训练并使用投票集成方法评估
    支持直接测试模式和约登指数评估
    
    参数:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        hp: 超参数字典
        exp_dir: 交叉验证结果目录
        num_runs: 训练次数
        direct_test: 是否直接测试模式
        model_dir: 预训练模型所在目录，用于直接测试模式
        
    返回:
        tuple: (最佳单模型路径, 集成模型目录)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果是直接测试模式，使用指定的模型目录
    if direct_test:
        model_dir = model_dir or exp_dir
        print(f"\n直接测试模式：从 {model_dir} 加载预训练模型")
        ensemble_dir = os.path.join(model_dir, "test_results")
    else:
        # 原有的训练模式
        # 1. 创建集成训练目录
        ensemble_dir = os.path.join(exp_dir, "ensemble_models")
    
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # 加载或训练模型
    model_objects = []
    model_paths = []
    all_metrics = []
    
    if direct_test:
        # 直接测试模式：加载预训练模型
        print("查找预训练模型...")
        
        # 查找模型种子目录
        seed_dirs = sorted(glob.glob(os.path.join(model_dir, "ensemble_models/model_seed_*")))
        if seed_dirs:
            for i, seed_dir in enumerate(seed_dirs):
                seed = os.path.basename(seed_dir).split("_")[-1]
                
                # 查找模型权重文件
                weight_path = os.path.join(seed_dir, "best_weights.pth")
                if os.path.exists(weight_path):
                    print(f"加载模型 {i+1}: 种子 {seed} from {weight_path}")
                    
                    # 加载模型
                    model = get_model(hp)
                    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                    model = model.to(device)
                    model.eval()
                    
                    model_objects.append(model)
                    model_paths.append(weight_path)
                else:
                    print(f"警告: 在种子目录 {seed_dir} 中未找到模型权重文件")
        
        if not model_objects:
            # 备用：尝试查找最佳单个模型
            best_model_path = os.path.join(model_dir, "ensemble_models/best_single_model.pth")
            if os.path.exists(best_model_path):
                print(f"未找到种子模型，使用最佳单个模型: {best_model_path}")
                model = get_model(hp)
                model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                model = model.to(device)
                model.eval()
                
                model_objects.append(model)
                model_paths.append(best_model_path)
            else:
                raise ValueError(f"在 {model_dir} 中未找到任何可用的预训练模型")
                
        print(f"成功加载 {len(model_objects)} 个预训练模型")
    else:
        # 原有训练模式代码
        # 2. 从交叉验证结果确定最优训练轮数
        optimal_epochs = determine_optimal_epochs(exp_dir, hp)
        
        # 3. 准备不同的随机种子 (使用质数以减少随机数周期性)
        base_seeds = [41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
        seeds = base_seeds[:num_runs]
        
        # 4. 保存训练配置
        with open(os.path.join(ensemble_dir, "ensemble_config.txt"), 'w') as f:
            f.write(f"集成模型训练配置\n")
            f.write(f"==============\n\n")
            f.write(f"训练次数: {num_runs}\n")
            f.write(f"最优训练轮数: {optimal_epochs}\n")
            f.write(f"随机种子: {seeds}\n\n")
        
        # 5. 多次训练
        for i, seed in enumerate(seeds):
            print(f"\n{'='*20} 训练 {i+1}/{num_runs} (种子: {seed}) {'='*20}")
            
            model_path, val_auc, test_auc = train_single_model(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                hp=hp,
                output_dir=ensemble_dir,
                seed=seed,
                optimal_epochs=optimal_epochs
            )
            
            # 加载模型用于集成预测
            model = get_model(hp)
            # 注意：确保先加载到CPU，避免cuda:0/cuda:1等设备不匹配问题
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            # 然后明确地将整个模型移动到当前活动设备
            model = model.to(device)
            model_objects.append(model)
            model_paths.append(model_path)
    
    # 准备测试数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hp["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 如果是训练模式，需要评估每个单独的模型
    if not direct_test:
        for i, (model, model_path) in enumerate(zip(model_objects, model_paths)):
            # 获取模型预测用于计算指标
            all_preds = []
            all_probs = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(float)
                    
                    all_preds.extend(preds)
                    all_probs.extend(probs)
                    all_labels.extend(labels.numpy().flatten())
            
            # 计算标准阈值下的完整指标集
            standard_metrics = calculate_metrics(all_preds, all_labels, all_probs)
            
            # 计算基于约登指数的最佳阈值和相应指标
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            
            # 计算约登指数 (J = 敏感度 + 特异度 - 1)
            specificity = 1 - fpr
            youden_index = tpr + specificity - 1
            
            # 找到最大约登指数对应的索引
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            
            # 使用最佳阈值重新生成预测标签
            youden_preds = [1 if p >= optimal_threshold else 0 for p in all_probs]
            
            # 计算约登指数优化后的指标
            youden_metrics = calculate_metrics(youden_preds, all_labels, all_probs)
            
            # 合并所有指标
            seed = os.path.basename(model_path).split("_")[1] if "_" in os.path.basename(model_path) else i
            metrics = {
                'standard': standard_metrics,
                'youden': youden_metrics,
                'optimal_threshold': optimal_threshold,
                'seed': seed,
                'model_path': model_path,
            }
            all_metrics.append(metrics)
            
            # 保存完整指标报告
            model_metrics_path = os.path.join(os.path.dirname(model_path), f"model_{seed}_metrics.txt")
            with open(model_metrics_path, 'w') as f:
                f.write(f"模型 {i+1} (种子 {seed}) 测试集指标\n")
                f.write("="*40 + "\n\n")
                
                f.write("标准阈值(0.5)下的指标:\n")
                f.write(f"  准确率: {standard_metrics['accuracy']*100:.2f}%\n")
                f.write(f"  AUC: {standard_metrics['auc']:.4f}\n")
                f.write(f"  F1分数: {standard_metrics['f1']:.4f}\n")
                f.write(f"  敏感度: {standard_metrics['sensitivity']:.4f}\n")
                f.write(f"  特异度: {standard_metrics['specificity']:.4f}\n\n")
                
                f.write(f"约登指数优化 (最佳阈值: {optimal_threshold:.4f}):\n")
                f.write(f"  准确率: {youden_metrics['accuracy']*100:.2f}%\n")
                f.write(f"  F1分数: {youden_metrics['f1']:.4f}\n")
                f.write(f"  敏感度: {youden_metrics['sensitivity']:.4f}\n")
                f.write(f"  特异度: {youden_metrics['specificity']:.4f}\n\n")
                
                f.write(f"混淆矩阵 (最佳阈值):\n{youden_metrics['conf_mat']}\n")
            
            # 绘制ROC曲线并标记最佳阈值点
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {standard_metrics["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            
            # 标记最佳阈值点
            plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                     label=f'最佳阈值: {optimal_threshold:.3f}\n'
                           f'敏感度: {youden_metrics["sensitivity"]:.3f}, '
                           f'特异度: {youden_metrics["specificity"]:.3f}')
            
            plt.axis([0, 1, 0, 1])
            plt.xlabel('假阳性率 (1 - 特异度)', fontsize=12)
            plt.ylabel('真阳性率 (敏感度)', fontsize=12)
            plt.title(f'模型 {i+1} (种子 {seed}) ROC曲线', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            roc_path = os.path.join(os.path.dirname(model_path), f"model_{seed}_roc.png")
            plt.savefig(roc_path)
            plt.close()
    
    # 如果是训练模式，汇总各个模型的性能
    if not direct_test and all_metrics:
        with open(os.path.join(ensemble_dir, "individual_models_summary.txt"), 'w') as f:
            f.write("个体模型性能汇总\n")
            f.write("==============\n\n")
            
            for i, metrics in enumerate(all_metrics):
                standard = metrics['standard']
                youden = metrics['youden']
                
                f.write(f"模型 {i+1}:\n")
                f.write(f"  随机种子: {metrics['seed']}\n")
                f.write(f"  路径: {metrics['model_path']}\n")
                f.write(f"  AUC: {standard['auc']:.4f}\n")
                f.write(f"  标准阈值下准确率: {standard['accuracy']*100:.2f}%\n")
                f.write(f"  最佳阈值: {metrics['optimal_threshold']:.4f}\n")
                f.write(f"  最佳阈值下准确率: {youden['accuracy']*100:.2f}%\n")
                f.write(f"  最佳阈值下敏感度: {youden['sensitivity']:.4f}\n")
                f.write(f"  最佳阈值下特异度: {youden['specificity']:.4f}\n\n")
        
        # 找出最佳单个模型 (基于AUC)
        best_model_idx = np.argmax([m['standard']['auc'] for m in all_metrics])
        best_model_metrics = all_metrics[best_model_idx]
        best_model_path = best_model_metrics['model_path']
        
        # 复制最佳单个模型到顶层目录
        import shutil
        best_copy_path = os.path.join(ensemble_dir, "best_single_model.pth")
        shutil.copy(best_model_path, best_copy_path)
        
        print(f"\n最佳单个模型: 模型 {best_model_idx+1} (种子 {best_model_metrics['seed']})")
        print(f"测试集 AUC: {best_model_metrics['standard']['auc']:.4f}")
        print(f"已复制到: {best_copy_path}")
    
    # 执行集成预测
    print("\n执行集成预测...")
    # all_predictions = []
    # all_probs = []
    # true_labels = None
    
    # for i, model in enumerate(model_objects):
    #     model.eval()
        
    #     model_preds = []
    #     model_probs = []
    #     labels_list = []
        
    #     with torch.no_grad():
    #         for images, labels in tqdm(test_loader, desc=f"预测 - 模型 {i+1}/{len(model_objects)}"):
    #             images = images.to(device)
    #             outputs = model(images)
    #             probs = torch.sigmoid(outputs).cpu().numpy().flatten()
    #             preds = (probs > 0.5).astype(float)
                
    #             model_preds.extend(preds)
    #             model_probs.extend(probs)
                
    #             if true_labels is None:
    #                 labels_list.extend(labels.numpy().flatten())
            
    #     all_predictions.append(model_preds)
    #     all_probs.append(model_probs)
        
    #     if true_labels is None:
    #         true_labels = labels_list
    
    # # 转换为numpy数组以便于操作
    # all_predictions = np.array(all_predictions)
    # all_probs = np.array(all_probs)
    
    # # 平均概率
    # ensemble_probs = np.mean(all_probs, axis=0)
    

    ensemble_preds, ensemble_probs, true_labels = ensemble_predictions(
        model_objects, 
        test_loader, 
        device,
        optimal_threshold=0.5
    )
    # 计算集成模型的标准评估指标（阈值0.5）
    standard_ensemble_metrics = calculate_metrics(ensemble_preds, true_labels, ensemble_probs)
    



    # 计算约登指数优化的指标
    # # 计算ROC曲线
    # fpr, tpr, thresholds = roc_curve(true_labels, ensemble_probs)


    # 准备测试数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hp["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    _, ensemble_probs_train, true_labels_train = ensemble_predictions(
        model_objects, 
        train_loader, 
        device,
        optimal_threshold=0.5
    )
    fpr, tpr, thresholds = roc_curve(true_labels_train, ensemble_probs_train)
    



    # 计算约登指数 (J = 敏感度 + 特异度 - 1)
    specificity = 1 - fpr
    youden_index = tpr + specificity - 1
    
    # 找到最大约登指数对应的索引
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    # 使用最佳阈值生成预测
    youden_ensemble_preds = (ensemble_probs > optimal_threshold).astype(float)
    # 计算约登指数优化后的指标
    youden_ensemble_metrics = calculate_metrics(youden_ensemble_preds, true_labels, ensemble_probs)
    with open(os.path.join(ensemble_dir, 'ensemble_probs.pkl'), 'wb') as f:
        pickle.dump({
            'ensemble_probs': ensemble_probs,
            'true_labels': true_labels
        }, f)
        
    # 保存集成模型标准评估报告
    standard_report_path = os.path.join(ensemble_dir, "ensemble_standard_metrics.txt")
    with open(standard_report_path, 'w') as f:
        f.write("集成模型测试集指标 (标准阈值0.5)\n")
        f.write("==========================\n\n")
        f.write(f"准确率: {standard_ensemble_metrics['accuracy']*100:.2f}%\n")
        # f.write(f"AUC: {standard_ensemble_metrics['auc']:.4f}\n")
        # 更新集成模型报告
        f.write(f"AUC: {standard_ensemble_metrics['auc']:.4f} (95% CI: {standard_ensemble_metrics['auc_ci_lower']:.2f}-{standard_ensemble_metrics['auc_ci_upper']:.2f})\n")
        f.write(f"F1分数: {standard_ensemble_metrics['f1']:.4f}\n")
        f.write(f"敏感度: {standard_ensemble_metrics['sensitivity']:.4f}\n")
        f.write(f"特异度: {standard_ensemble_metrics['specificity']:.4f}\n\n")
        
        f.write(f"混淆矩阵:\n{standard_ensemble_metrics['conf_mat']}\n")
    
    # 保存约登指数优化的集成模型评估报告
    youden_report_path = os.path.join(ensemble_dir, "ensemble_youden_metrics.txt")
    with open(youden_report_path, 'w') as f:
        f.write("集成模型测试集指标 (约登指数优化)\n")
        f.write("==========================\n\n")
        f.write(f"最佳阈值: {optimal_threshold:.4f}\n\n")
        f.write(f"准确率: {youden_ensemble_metrics['accuracy']*100:.2f}%\n")
        # f.write(f"AUC: {youden_ensemble_metrics['auc']:.4f}\n")
        # 更新集成模型报告
        f.write(f"AUC: {youden_ensemble_metrics['auc']:.4f} (95% CI: {youden_ensemble_metrics['auc_ci_lower']:.2f}-{youden_ensemble_metrics['auc_ci_upper']:.2f})\n")
        f.write(f"F1分数: {youden_ensemble_metrics['f1']:.4f}\n")
        f.write(f"敏感度: {youden_ensemble_metrics['sensitivity']:.4f}\n")
        f.write(f"特异度: {youden_ensemble_metrics['specificity']:.4f}\n\n")
        
        f.write(f"混淆矩阵:\n{youden_ensemble_metrics['conf_mat']}\n")
    
    # 绘制集成模型ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {standard_ensemble_metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # 标记最佳阈值点
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
             label=f'最佳阈值: {optimal_threshold:.3f}\n'
                   f'敏感度: {youden_ensemble_metrics["sensitivity"]:.3f}, '
                   f'特异度: {youden_ensemble_metrics["specificity"]:.3f}')
    
    plt.axis([0, 1, 0, 1])
    plt.xlabel('假阳性率 (1 - 特异度)', fontsize=12)
    plt.ylabel('真阳性率 (敏感度)', fontsize=12)
    plt.title('集成模型 ROC 曲线', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    roc_path = os.path.join(ensemble_dir, "ensemble_roc.png")
    plt.savefig(roc_path)
    plt.close()
    
    # 如果是训练模式且有单个模型评估数据，则比较单个模型与集成模型性能
    if not direct_test and all_metrics:
        best_standard = best_model_metrics['standard']
        best_youden = best_model_metrics['youden']
        
        with open(os.path.join(ensemble_dir, "comparison_report.txt"), 'w') as f:
            f.write("个体模型与集成模型性能比较\n")
            f.write("====================\n\n")
            
            # 计算平均指标
            avg_acc = np.mean([m['standard']['accuracy'] for m in all_metrics])
            avg_auc = np.mean([m['standard']['auc'] for m in all_metrics])
            avg_f1 = np.mean([m['standard']['f1'] for m in all_metrics])
            std_auc = np.std([m['standard']['auc'] for m in all_metrics])
            
            f.write(f"单个模型平均指标:\n")
            f.write(f"  准确率: {avg_acc*100:.2f}%\n")
            f.write(f"  AUC: {avg_auc:.4f} ± {std_auc:.4f}\n")
            f.write(f"  F1分数: {avg_f1:.4f}\n\n")
            
            f.write(f"最佳单个模型 (种子 {best_model_metrics['seed']}):\n")
            f.write(f"  标准阈值下:\n")
            f.write(f"    准确率: {best_standard['accuracy']*100:.2f}%\n")
            f.write(f"    AUC: {best_standard['auc']:.4f}\n")
            f.write(f"    F1分数: {best_standard['f1']:.4f}\n")
            f.write(f"    敏感度: {best_standard['sensitivity']:.4f}\n")
            f.write(f"    特异度: {best_standard['specificity']:.4f}\n\n")
            
            f.write(f"  最佳阈值({best_model_metrics['optimal_threshold']:.4f})下:\n")
            f.write(f"    准确率: {best_youden['accuracy']*100:.2f}%\n")
            f.write(f"    F1分数: {best_youden['f1']:.4f}\n")
            f.write(f"    敏感度: {best_youden['sensitivity']:.4f}\n")
            f.write(f"    特异度: {best_youden['specificity']:.4f}\n\n")
            
            f.write(f"集成模型:\n")
            f.write(f"  标准阈值下:\n")
            f.write(f"    准确率: {standard_ensemble_metrics['accuracy']*100:.2f}%\n")
            f.write(f"    AUC: {standard_ensemble_metrics['auc']:.4f}\n")
            f.write(f"    F1分数: {standard_ensemble_metrics['f1']:.4f}\n")
            f.write(f"    敏感度: {standard_ensemble_metrics['sensitivity']:.4f}\n")
            f.write(f"    特异度: {standard_ensemble_metrics['specificity']:.4f}\n\n")
            
            f.write(f"  最佳阈值({optimal_threshold:.4f})下:\n")
            f.write(f"    准确率: {youden_ensemble_metrics['accuracy']*100:.2f}%\n")
            f.write(f"    F1分数: {youden_ensemble_metrics['f1']:.4f}\n")
            f.write(f"    敏感度: {youden_ensemble_metrics['sensitivity']:.4f}\n")
            f.write(f"    特异度: {youden_ensemble_metrics['specificity']:.4f}\n\n")
            
            # 计算提升（使用约登指数优化的结果）
            acc_lift = youden_ensemble_metrics['accuracy'] - best_youden['accuracy']
            auc_lift = standard_ensemble_metrics['auc'] - best_standard['auc']  # AUC不受阈值影响
            f1_lift = youden_ensemble_metrics['f1'] - best_youden['f1']
            
            f.write(f"相比最佳单个模型的提升(基于最佳阈值):\n")
            f.write(f"  准确率: {acc_lift*100:.2f}%\n")
            f.write(f"  AUC: {auc_lift:.4f}\n")
            f.write(f"  F1分数: {f1_lift:.4f}\n")
    
    # 控制台输出比较结果
    print("\n" + "="*50)
    print("集成模型最终评估结果")
    print("="*50)
    print(f"标准阈值(0.5)下:")
    print(f"  准确率: {standard_ensemble_metrics['accuracy']*100:.2f}%")
    print(f"  AUC: {standard_ensemble_metrics['auc']:.4f} (95% CI: {standard_ensemble_metrics['auc_ci_lower']:.2f}-{standard_ensemble_metrics['auc_ci_upper']:.2f})")
    print(f"  F1分数: {standard_ensemble_metrics['f1']:.4f}")
    print(f"  敏感度: {standard_ensemble_metrics['sensitivity']:.4f}")
    print(f"  特异度: {standard_ensemble_metrics['specificity']:.4f}")
    
    print(f"\n约登指数优化 (最佳阈值: {optimal_threshold:.4f}):")
    print(f"  准确率: {youden_ensemble_metrics['accuracy']*100:.2f}%")
    print(f"  F1分数: {youden_ensemble_metrics['f1']:.4f}")
    print(f"  敏感度: {youden_ensemble_metrics['sensitivity']:.4f}")
    print(f"  特异度: {youden_ensemble_metrics['specificity']:.4f}")
    
    if not direct_test and all_metrics:
        print(f"\n相比最佳单个模型 (种子 {best_model_metrics['seed']}):")
        print(f"  AUC提升: {auc_lift:.4f}")
        print(f"  准确率提升(最佳阈值): {acc_lift*100:.2f}%")
    
    # 返回最佳单模型路径和集成目录
    best_copy_path = best_copy_path if not direct_test and all_metrics else None
    return best_copy_path, ensemble_dir


def ensemble_cv_models_on_test(exp_dir, test_dataset, hp):
    """将交叉验证中的所有最佳模型根据权重集成起来进行测试"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hp["batch_size"], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载所有模型
    models = []
    model_weights = []
    model_types = []  # 记录模型类型（ACC或AUC优化）
    fold_dirs = [os.path.join(exp_dir, f"fold_{i+1}") for i in range(hp["n_splits"])]
    
    # 加载ACC优化的模型
    for fold, fold_dir in enumerate(fold_dirs):
        best_model_path = os.path.join(fold_dir, "best_model_acc.pth")
        if os.path.exists(best_model_path):
            print(f"加载 Fold {fold+1} 的准确率优化模型 ({best_model_path})")
            model = get_model(hp).to(device)
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 从检查点获取验证准确率作为权重
            weight = checkpoint.get('val_acc', 0.5)  # 使用验证准确率作为权重
            
            model.eval()
            models.append(model)
            model_weights.append(weight)
            model_types.append("ACC")
            print(f"ACC模型权重: {weight:.4f} (基于验证准确率)")
        else:
            print(f"警告: 找不到 Fold {fold+1} 的准确率优化模型文件")

    # 加载AUC优化的模型
    for fold, fold_dir in enumerate(fold_dirs):
        best_model_path = os.path.join(fold_dir, "best_model_auc.pth")
        if os.path.exists(best_model_path):
            print(f"加载 Fold {fold+1} 的AUC优化模型 ({best_model_path})")
            model = get_model(hp).to(device)
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 从检查点获取验证AUC作为权重
            weight = checkpoint.get('val_auc', 0.5)  # 使用验证AUC作为权重
            
            model.eval()
            models.append(model)
            model_weights.append(weight)
            model_types.append("AUC")
            print(f"AUC模型权重: {weight:.4f} (基于验证AUC)")

    if not models:
        print("错误: 没有找到任何模型，无法进行集成评估")
        return 0, 0, 0, None
    
    # 分别归一化ACC和AUC模型的权重
    # 这样两类模型可以有相似的总体贡献
    acc_indices = [i for i, t in enumerate(model_types) if t == "ACC"]
    auc_indices = [i for i, t in enumerate(model_types) if t == "AUC"]
    
    if acc_indices:
        acc_weights_sum = sum(model_weights[i] for i in acc_indices)
        for i in acc_indices:
            if acc_weights_sum > 0:
                model_weights[i] = model_weights[i] / acc_weights_sum * 0.5  # ACC模型总权重为0.5
            else:
                model_weights[i] = 0.5 / len(acc_indices)  # 平均分配0.5权重
    
    if auc_indices:
        auc_weights_sum = sum(model_weights[i] for i in auc_indices)
        for i in auc_indices:
            if auc_weights_sum > 0:
                model_weights[i] = model_weights[i] / auc_weights_sum * 0.5  # AUC模型总权重为0.5
            else:
                model_weights[i] = 0.5 / len(auc_indices)  # 平均分配0.5权重
    
    print("\n归一化后的模型权重:")
    for i, (w, t) in enumerate(zip(model_weights, model_types)):
        print(f"模型 {i+1} ({t}): {w:.4f}")
    
    # 创建损失函数
    if hp["use_balanced_loss"]:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hp["pos_weight"]]).to(device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 收集所有预测结果
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估加权集成模型"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            
            # 收集所有模型的预测
            batch_probs = []
            for i, model in enumerate(models):
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                batch_probs.append(probs)
            
            # 使用权重平均所有模型的预测概率
            batch_probs = np.array(batch_probs)  # shape: (n_models, batch_size)
            weighted_probs = np.sum(batch_probs * np.array(model_weights)[:, np.newaxis], axis=0)
            all_probs.extend(weighted_probs.tolist())
    
    # 生成预测标签和计算指标
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    test_loss = running_loss / len(test_loader.dataset)
    
    correct = sum([1 for p, t in zip(all_preds, all_labels) if p == t])
    test_acc = correct / len(all_labels)
    
    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        test_auc = 0.0
    
    test_f1 = f1_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    print("\n加权集成模型在测试集上的性能:")
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc*100:.2f}%")
    print(f"测试集AUC: {test_auc:.4f}")
    print(f"测试集F1: {test_f1:.4f}")
    print(f"混淆矩阵:\n{conf_mat}")
    
    # 计算混淆矩阵中的具体指标
    tn, fp, fn, tp = conf_mat.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"灵敏度(召回率): {sensitivity:.4f}")
    print(f"特异度: {specificity:.4f}")
    print(f"阳性预测值(精确率): {ppv:.4f}")
    print(f"阴性预测值: {npv:.4f}")
    
    # 绘制ROC曲线并保存
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Weighted Ensemble Model ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    roc_path = os.path.join(exp_dir, "weighted_ensemble_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC曲线已保存至: {roc_path}")
    
    return test_acc, test_auc, test_f1, conf_mat

# # ================ 13) 全部数据训练函数 ===================
# def train_full_dataset(train_dataset, test_dataset, hp):
#     """
#     使用全部训练数据训练，在测试集上评估
    
#     参数:
#         train_dataset: 完整训练数据集
#         test_dataset: 测试数据集
#         hp: 超参数字典
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
    
#     # 准备结果目录
#     results_dir = hp["results_dir"]
#     experiment_name = f"resnet{hp['model_depth']}_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     exp_dir = os.path.join(results_dir, experiment_name)
#     os.makedirs(exp_dir, exist_ok=True)
    
#     # 保存超参数
#     with open(os.path.join(exp_dir, "hyperparameters.txt"), 'w') as f:
#         for key, value in hp.items():
#             f.write(f"{key}: {value}\n")
    
#     # 创建数据加载器
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=hp["batch_size"], 
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=hp["batch_size"], 
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # 创建模型
#     model = get_model(hp).to(device)
    
#     # 创建损失函数
#     if hp["use_balanced_loss"]:
#         criterion = nn.BCEWithLogitsLoss(
#             pos_weight=torch.tensor([hp["pos_weight"]]).to(device)
#         )
#     else:
#         criterion = nn.BCEWithLogitsLoss()
    
#     # 创建优化器
#     optimizer = get_optimizer_with_layerwise_lr(model, hp)
    
#     # 创建学习率调度器
#     scheduler = build_scheduler(optimizer, hp)
    
#     # 创建解冻调度器
#     unfreeze_scheduler = None
#     if hp["progressive_unfreeze"] and hp["freeze_layers"]:
#         if hp.get("use_dynamic_unfreezing", False):
#             # 定义解冻层的分组（从上到下）
#             layer_groups = [
#                 ['mri_layer3.5', 'mri_layer3.4', 'mask_layer3.5', 'mask_layer3.4'],
#                 ['mri_layer3.3', 'mri_layer3.2', 'mask_layer3.3', 'mask_layer3.2'],
#                 ['mri_layer3.1', 'mri_layer3.0', 'mask_layer3.1', 'mask_layer3.0'],
#                 ['mri_layer2.3', 'mri_layer2.2', 'mask_layer2.3', 'mask_layer2.2'],
#                 ['mri_layer2.1', 'mri_layer2.0', 'mask_layer2.1', 'mask_layer2.0'],
#                 ['mri_layer1.1', 'mri_layer1.0', 'mask_layer1.1', 'mask_layer1.0'],
#                 ['mri_conv1', 'mri_bn1', 'mask_conv1', 'mask_bn1']
#             ]
            
#             unfreeze_scheduler = DynamicUnfreezing(
#                 model=model,
#                 layer_groups=layer_groups,
#                 monitor=hp.get("unfreeze_monitor", "val_acc"),
#                 plateau_patience=hp.get("plateau_patience", 3),
#                 min_delta=hp.get("unfreeze_min_delta", 0.002),
#                 min_epochs_between_unfreeze=hp.get("min_epochs_between_unfreeze", 3)
#             )
#         else:
#             # 基于epoch的解冻
#             unfreeze_scheduler = ProgressiveUnfreezing(
#                 model, 
#                 unfreeze_epoch_map=hp["unfreeze_epoch_map"]
#             )
    
#     # 训练模型
#     best_acc, best_auc = train_one_fold(
#         fold=0,  # 用0表示全部数据训练
#         train_loader=train_loader,
#         val_loader=test_loader,  # 使用测试集作为验证集
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         hp=hp,
#         fold_dir=exp_dir,
#         unfreeze_scheduler=unfreeze_scheduler
#     )
    
#     print("\n" + "="*50)
#     print(f"全部数据训练完成 - {experiment_name}")
#     print(f"最佳测试集准确率: {best_acc*100:.2f}%")
#     print(f"最佳测试集AUC: {best_auc:.4f}")
#     print("="*50)
    
#     return best_acc, best_auc

# ================ 14) 主程序 ===================
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # ----------- 准备数据集 -----------
    # 根据您的实际情况修改以下路径
    base_path = "/home/vipuser/Desktop/Data/Task02_PASp62_edge"
    
    # =========================
    # 1) 训练数据
    # =========================
    # 排序文件名
    train_image_filenames = sorted(os.listdir(os.path.join(base_path, 'imagesTr')))
    train_label_filenames = sorted(os.listdir(os.path.join(base_path, 'labelsTr')))
    
    # 构建路径
    train_image_paths = [
        os.path.join(base_path, 'imagesTr', file) for file in train_image_filenames
    ]
    train_label_paths = [
        os.path.join(base_path, 'labelsTr', file) for file in train_label_filenames
    ]
    
    # 构建诊断标签
    train_diag_labels = []
    for label_file in train_label_filenames:
        file_identifier = label_file[:-len(".nii.gz")]  # 去除".nii.gz"
        no_pas_match = glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file_identifier + '*.nii.gz')
        ) or glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file_identifier + '*.nii.gz')
        )
        pas_match = glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file_identifier + '*.nii.gz')
        ) or glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file_identifier + '*.nii.gz')
        )
        
        if no_pas_match:
            train_diag_labels.append(0)
        elif pas_match:
            train_diag_labels.append(1)
        else:
            raise RuntimeError(f"无法在PAS/NoPAS目录中找到文件 {label_file}")
    
    # =========================
    # 2) 测试数据
    # =========================
    # 排序文件名
    test_image_filenames = sorted(os.listdir(os.path.join(base_path, 'imagesTs')))
    test_label_filenames = sorted(os.listdir(os.path.join(base_path, 'labelsTs')))
    
    # 构建路径
    test_image_paths = [
        os.path.join(base_path, 'imagesTs', file) for file in test_image_filenames
    ]
    test_label_paths = [
        os.path.join(base_path, 'labelsTs', file) for file in test_label_filenames
    ]
    
    # 构建诊断标签
    test_diag_labels = []
    for label_file in test_label_filenames:
        file_identifier = label_file[:-len(".nii.gz")]
        no_pas_match = glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/NoPAS', file_identifier + '*.nii.gz')
        ) or glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/NoPAS', file_identifier + '*.nii.gz')
        )
        pas_match = glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center1/PAS', file_identifier + '*.nii.gz')
        ) or glob.glob(
            os.path.join('/home/vipuser/Desktop/Data/PASRawData/center2/PAS', file_identifier + '*.nii.gz')
        )
        
        if no_pas_match:
            test_diag_labels.append(0)
        elif pas_match:
            test_diag_labels.append(1)
        else:
            raise RuntimeError(f"无法在PAS/NoPAS目录中找到文件 {label_file}")
    
    # 创建数据集
    print("创建训练数据集...")
    full_train_dataset = MRIDataset3D(
        train_image_paths,
        train_label_paths,
        train_diag_labels,
        is_train=False,
        fixed_depth=HYPERPARAMS["fixed_depth"]
    )
    
    print("创建测试数据集...")
    test_dataset = MRIDataset3D(
        test_image_paths,
        test_label_paths,
        test_diag_labels,
        is_train=False,
        fixed_depth=HYPERPARAMS["fixed_depth"]
    )
    
    # 打印数据集信息
    print(f"训练数据集大小: {len(full_train_dataset)}")
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 统计类别分布
    train_pos = sum(train_diag_labels)
    train_neg = len(train_diag_labels) - train_pos
    test_pos = sum(test_diag_labels)
    test_neg = len(test_diag_labels) - test_pos
    
    print(f"训练集类别分布: 正例 {train_pos}个 ({train_pos/len(train_diag_labels)*100:.1f}%), "
          f"负例 {train_neg}个 ({train_neg/len(train_diag_labels)*100:.1f}%)")
    print(f"测试集类别分布: 正例 {test_pos}个 ({test_pos/len(test_diag_labels)*100:.1f}%), "
          f"负例 {test_neg}个 ({test_neg/len(test_diag_labels)*100:.1f}%)")
    
    # ============ 选择训练模式 ============
    
    # 1. 交叉验证模式
    print("\n开始5折交叉验证训练...")
    cross_validate(
        train_dataset=full_train_dataset,
        hp=HYPERPARAMS,
        test_dataset=test_dataset,  # 添加测试集参数
        # only_test_dataset='/home/vipuser/Desktop/test_new_file/edge/resnet18_20250402_044228'
        # only_test_dataset = '/home/vipuser/Desktop/test_new_file/edge/resnet18_20250403_013532_SingleBranch'
        # only_test_dataset='/home/vipuser/Desktop/test_new_file1/resnet18_20250423_134608'
        # only_test_dataset='/home/vipuser/Desktop/test_new_file1/resnet18_20250422_150738'
        # only_test_dataset='/home/vipuser/Desktop/test_new_file2/edge/resnet18_20250424_170121'
        only_test_dataset='/home/vipuser/Desktop/test_new_file2/edge/resnet18_20250424_212947'
    )
    
    # # 2. 全部数据训练模式（可选，取消注释使用）
    # print("\n使用所有训练数据进行训练...")
    # train_full_dataset(
    #     train_dataset=full_train_dataset,
    #     test_dataset=test_dataset,
    #     hp=HYPERPARAMS
    # )
    
    print("\n训练完成！")