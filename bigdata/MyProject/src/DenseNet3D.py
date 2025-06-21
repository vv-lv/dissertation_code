import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
# CBAMBlock3D：3D卷积注意力模块（通道+空间）
# ---------------------------------------------------
class CBAMBlock3D(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=5):
        super(CBAMBlock3D, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1) 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 2) 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        
        return x


# ---------------------------------------------------
# Branch3DExtractor：基础 3D 卷积分支
# ---------------------------------------------------
class Branch3DExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(Branch3DExtractor, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# ---------------------------------------------------
# _DenseLayer3D：单层 Dense 层
# ---------------------------------------------------
class _DenseLayer3D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer3D, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return torch.cat([x, new_features], 1)


# ---------------------------------------------------
# _DenseBlock3D：多个 Dense 层堆叠
# ---------------------------------------------------
class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlock3D, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------
# _Transition3D：过渡层(降维 + 池化)
# ---------------------------------------------------
class _Transition3D(nn.Module):
    def __init__(self, num_input_features, num_output_features,
                 pool_kernel=(2, 2, 2), pool_stride=(2, 2, 2)):
        super(_Transition3D, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


# ---------------------------------------------------
# DenseNet3DCBAM：带 CBAM 的 3D DenseNet 主体
# ---------------------------------------------------
class DenseNet3DCBAM(nn.Module):
    def __init__(self, 
                 in_channels=64,
                 num_classes=1,
                 growth_rate=12,
                 block_config=(2, 4, 6, 4),
                 bn_size=4,
                 drop_rate=0.4,
                 num_init_features=32):
        super(DenseNet3DCBAM, self).__init__()

        # 初始卷积 -> BN -> ReLU -> MaxPool
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features,
                      kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # 逐个构建 Dense Block + CBAM + Transition
        for i, num_layers in enumerate(block_config):
            dense_block = _DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.dense_blocks.append(dense_block)
            num_features = num_features + num_layers * growth_rate

            # CBAM
            self.cbam_blocks.append(CBAMBlock3D(num_features))

            # Transition(除了最后一个Block都加过渡层)
            if i != len(block_config) - 1:
                if i == len(block_config) - 2:
                    transition = _Transition3D(
                        num_input_features=num_features,
                        num_output_features=num_features // 2,
                        pool_kernel=(1, 2, 2),
                        pool_stride=(1, 2, 2)
                    )
                else:
                    transition = _Transition3D(
                        num_input_features=num_features,
                        num_output_features=num_features // 2
                    )
                self.transitions.append(transition)
                num_features = num_features // 2

        # 最后的 dropout + BN + ReLU + 全局池化 + 全连接
        self.dropout_final = nn.Dropout(p=drop_rate)
        self.bn_final = nn.BatchNorm3d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ============== 新增方法 ==============
    def forward_feature(self, x):
        """
        只走到 Global Pool + Flatten，返回特征 (B, num_features)
        """
        x = self.features(x)
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            x = self.cbam_blocks[i](x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x
    # ============== /新增方法 ==============

    def forward(self, x):
        x = self.features(x)
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            x = self.cbam_blocks[i](x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        print(x.shape)

        x = self.dropout_final(x)
        x = self.classifier(x)
        return x


class DenseNet3DParallelBranch(nn.Module):
    def __init__(self, 
                 num_classes=1,
                 growth_rate=8,  # 降低的增长率
                 block_config=(1, 2, 2, 1),  # 对称简化的块配置
                 bn_size=4,
                 drop_rate=0.4,
                 num_init_features=32):
        super(DenseNet3DParallelBranch, self).__init__()
        
        # MRI 分支的前半部分 (初始特征提取)
        self.mri_features = nn.Sequential(
            nn.Conv3d(1, num_init_features,
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # 掩码分支的前半部分 (与MRI分支结构相同但参数独立)
        self.mask_features = nn.Sequential(
            nn.Conv3d(1, num_init_features,
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # MRI分支的DenseBlock-1和Transition-1
        self.mri_dense_block1 = self._make_dense_block(_DenseBlock3D, num_init_features, 
                                                      block_config[0], growth_rate, bn_size, drop_rate)
        num_features = num_init_features + block_config[0] * growth_rate
        self.mri_transition1 = _Transition3D(num_features, num_features // 2)
        num_features = num_features // 2
        
        # MRI分支的DenseBlock-2和Transition-2
        self.mri_dense_block2 = self._make_dense_block(_DenseBlock3D, num_features, 
                                                      block_config[1], growth_rate, bn_size, drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.mri_transition2 = _Transition3D(num_features, num_features // 2)
        num_features = num_features // 2
        
        # 掩码分支的DenseBlock-1和Transition-1
        self.mask_dense_block1 = self._make_dense_block(_DenseBlock3D, num_init_features, 
                                                       block_config[0], growth_rate, bn_size, drop_rate)
        mask_num_features = num_init_features + block_config[0] * growth_rate
        self.mask_transition1 = _Transition3D(mask_num_features, mask_num_features // 2)
        mask_num_features = mask_num_features // 2
        
        # 掩码分支的DenseBlock-2和Transition-2
        self.mask_dense_block2 = self._make_dense_block(_DenseBlock3D, mask_num_features, 
                                                       block_config[1], growth_rate, bn_size, drop_rate)
        mask_num_features = mask_num_features + block_config[1] * growth_rate
        self.mask_transition2 = _Transition3D(mask_num_features, mask_num_features // 2)
        mask_num_features = mask_num_features // 2
        
        # 确保MRI和掩码分支输出的特征数量相同
        assert num_features == mask_num_features, "MRI和掩码分支的特征数量必须匹配才能融合"
        
        # 添加CBAM注意力模块到融合后的特征上
        self.fusion_cbam = CBAMBlock3D(num_features, reduction=4)  # 进一步减少CBAM复杂度
        
        # 共享的后半部分 (第3和第4个Dense块)
        self.shared_dense_block3 = self._make_dense_block(_DenseBlock3D, num_features, 
                                                         block_config[2], growth_rate, bn_size, drop_rate)
        num_features = num_features + block_config[2] * growth_rate
        self.shared_transition3 = _Transition3D(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.shared_dense_block4 = self._make_dense_block(_DenseBlock3D, num_features, 
                                                         block_config[3], growth_rate, bn_size, drop_rate)
        num_features = num_features + block_config[3] * growth_rate
        
        # 最后的BN、ReLU、池化和分类器
        self.norm_final = nn.BatchNorm3d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout_final = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_dense_block(self, block, num_input_features, num_layers, growth_rate, bn_size, drop_rate):
        """创建密集块的辅助方法"""
        return block(
            num_layers=num_layers,
            num_input_features=num_input_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            drop_rate=drop_rate
        )
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward_feature(self, x):
        """特征提取，返回(B, num_features)的张量"""
        # 分离输入数据
        x_mri = x[:, 0:1, ...]   # MRI通道
        x_mask = x[:, 1:2, ...]  # 掩码通道
        
        # MRI分支前向传播
        x_mri = self.mri_features(x_mri)
        x_mri = self.mri_dense_block1(x_mri)
        x_mri = self.mri_transition1(x_mri)
        x_mri = self.mri_dense_block2(x_mri)
        x_mri = self.mri_transition2(x_mri)
        
        # 掩码分支前向传播
        x_mask = self.mask_features(x_mask)
        x_mask = self.mask_dense_block1(x_mask)
        x_mask = self.mask_transition1(x_mask)
        x_mask = self.mask_dense_block2(x_mask)
        x_mask = self.mask_transition2(x_mask)
        
        # 特征融合 (这里使用加法而不是拼接，类似于ResNet示例)
        x_fused = x_mri + x_mask
        
        # # 应用CBAM注意力到融合特征
        # x_fused = self.fusion_cbam(x_fused)
        
        # 共享后半部分前向传播
        x_fused = self.shared_dense_block3(x_fused)
        x_fused = self.shared_transition3(x_fused)
        x_fused = self.shared_dense_block4(x_fused)
        
        # 最终处理
        x_fused = self.norm_final(x_fused)
        x_fused = self.relu_final(x_fused)
        x_fused = self.global_avg_pool(x_fused)
        x_fused = torch.flatten(x_fused, 1)
        
        return x_fused
    
    def forward(self, x):
        features = self.forward_feature(x)
        features = self.dropout_final(features)
        logits = self.classifier(features)
        return logits
    



class DenseNet3DMRIOnlyParallelInput(nn.Module):
    """
    仅使用 MRI 通道进行处理的 DenseNet3D 单分支模型。
    输入形状为 (B, 2, D, H, W)，但在 forward 中只使用 x[:, 0:1, ...] (MRI通道)。
    """
    def __init__(self, 
                 num_classes=1,
                 growth_rate=8,
                 block_config=(1, 2, 2, 1),  # 轻量级配置
                 bn_size=4,
                 drop_rate=0.4,
                 num_init_features=32):
        super(DenseNet3DMRIOnlyParallelInput, self).__init__()
        
        # 初始特征提取
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features,  # 注意这里输入通道是1
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # 跟踪当前特征数
        num_features = num_init_features
        
        # 实现四个Dense块和三个过渡层
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        # 构建Dense块和过渡层
        for i, num_layers in enumerate(block_config):
            # 添加Dense块
            block = _DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            
            # 添加CBAM注意力到每个Dense块后
            self.dense_blocks.append(CBAMBlock3D(num_features, reduction=4))
            
            # 在最后一个块之后不添加过渡层
            if i != len(block_config) - 1:
                # 创建过渡层，减少通道数
                trans = _Transition3D(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.transitions.append(trans)
                num_features = num_features // 2
        
        # 最后的处理层
        self.norm_final = nn.BatchNorm3d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout_final = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward_feature(self, x):
        """
        提取特征 (B, num_features)，不经过 classifier
        x 形状: (B, 2, D, H, W)
        """
        # 只取 MRI 通道 (索引 0)
        x_mri = x[:, 0:1, ...]  
        
        # 初始特征提取
        x = self.features(x_mri)
        
        # 依次通过所有Dense块和过渡层
        transition_idx = 0
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            # 过渡层只在CBAM后添加，且不在最后一个Dense块后添加
            if i % 2 == 1 and transition_idx < len(self.transitions):  # CBAM后才是过渡层
                x = self.transitions[transition_idx](x)
                transition_idx += 1
        
        # 最终处理
        x = self.norm_final(x)
        x = self.relu_final(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x):
        """
        前向传播
        x 形状: (B, 2, D, H, W)
        """
        features = self.forward_feature(x)
        features = self.dropout_final(features)
        logits = self.classifier(features)
        return logits
# # ---------------------------------------------------
# # 1) 多通道融合模型：DenseNet3DTwoChannelFusion
# # ---------------------------------------------------
# class DenseNet3DTwoChannelFusion(nn.Module):
#     """
#     演示：将 MRI 和分割掩码各走一个分支，然后拼接通道送入 DenseNet3D CBAM。
#     （原先的 "MILFusion" 可以重命名为这个）
#     """
#     def __init__(self, num_classes=1, growth_rate=12):  # 降低增长率
#         super(DenseNet3DTwoChannelFusion, self).__init__()
        
#         # 减少分支输出通道数
#         self.branch_mri = Branch3DExtractor(in_channels=1, out_channels=16)  # 从32减至16
#         self.branch_mask = Branch3DExtractor(in_channels=1, out_channels=16)  # 从32减至16
        
#         # 使用简化后的DenseNet3DCBAM
#         self.backbone = DenseNet3DCBAM(
#             in_channels=32,  # 对应两个分支各16通道
#             num_classes=num_classes,
#             growth_rate=growth_rate,
#             block_config=(2, 4, 6, 4),  # 简化块配置
#             bn_size=4,
#             drop_rate=0.4,  # 增加dropout
#             num_init_features=32  # 减少初始特征
#         )

#     # ============== 新增方法 ==============
#     def forward_feature(self, x):
#         """
#         提取特征 (B, num_features)，不经过 classifier
#         """
#         # 期望输入 x 形状: (B, 2, D, H, W)
#         x_mri = x[:, 0:1, ...]   # MRI
#         x_mask = x[:, 1:2, ...]  # Mask
        
#         out_mri = self.branch_mri(x_mri)     # (B, 32, D/2, H/2, W/2)
#         out_mask = self.branch_mask(x_mask)  # (B, 32, D/2, H/2, W/2)
        
#         fused = torch.cat([out_mri, out_mask], dim=1)  # (B, 64, ...)
#         feat = self.backbone.forward_feature(fused)  # 调用 DenseNet3DCBAM 的 forward_feature
#         return feat
#     # ============== /新增方法 ==============

#     def forward(self, x):
#         # 期望输入 x 形状: (B, 2, D, H, W)
#         x_mri = x[:, 0:1, ...]   # MRI
#         x_mask = x[:, 1:2, ...]  # Mask
        
#         out_mri = self.branch_mri(x_mri)     # (B, 32, D/2, H/2, W/2)
#         out_mask = self.branch_mask(x_mask)  # (B, 32, D/2, H/2, W/2)
        
#         fused = torch.cat([out_mri, out_mask], dim=1)  # (B, 64, ...)
#         out = self.backbone(fused)
#         return out


# # ---------------------------------------------------
# # 2) 单分支版本：但输入依旧是 (B, 2, D, H, W)，只取MRI通道
# # ---------------------------------------------------
# class DenseNet3DMRIOnlyTwoChannelInput(nn.Module):
#     """
#     仅使用 MRI 通道进行训练，忽略另一个 Mask 通道。
#     输入形状依然是 (B, 2, D, H, W)，但在 forward 中只截取 x[:, 0:1, ...]。
#     """
#     def __init__(self, num_classes=1, growth_rate=32):
#         super(DenseNet3DMRIOnlyTwoChannelInput, self).__init__()
        
#         # 只需要一个分支即可处理 MRI 通道
#         self.branch_mri = Branch3DExtractor(in_channels=1, out_channels=32)
        
#         # 单分支输出 32 通道，所以 DenseNet3DCBAM 的 in_channels=32
#         self.backbone = DenseNet3DCBAM(
#             in_channels=32,
#             num_classes=num_classes,
#             growth_rate=growth_rate,
#             block_config=(6, 12, 24, 16),
#             bn_size=4,
#             drop_rate=0.2,
#             num_init_features=64
#         )

#     # ============== 新增方法 ==============
#     def forward_feature(self, x):
#         """
#         提取特征 (B, num_features)，不经过 classifier
#         """
#         x_mri = x[:, 0:1, ...]    # 只取第0通道(MRI)
#         out_mri = self.branch_mri(x_mri)     
#         feat = self.backbone.forward_feature(out_mri)  # 调用 DenseNet3DCBAM 的 forward_feature
#         return feat
#     # ============== /新增方法 ==============

#     def forward(self, x):
#         """
#         x 形状: (B, 2, D, H, W)
#         这里我们只使用第 0 通道(MRI)，忽略第 1 通道(Mask)
#         """
#         x_mri = x[:, 0:1, ...]    # 仅取 MRI 通道
#         out_mri = self.branch_mri(x_mri)      # (B, 32, ...)
#         out = self.backbone(out_mri)          # (B, num_classes)
#         return out


# -------------------------------
# 主训练、验证与推理流程
# -------------------------------
if __name__ == '__main__':
    
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    from tqdm import tqdm

    # 设置设备：优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数配置
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-4
    step_size = 5   # 每 5 个 epoch 衰减一次学习率
    gamma = 0.5

    # 生成虚拟数据，数据形状符合模型要求：(batch, 2, 32, 128, 128)
    X_train = torch.randn(100, 2, 32, 128, 128)
    y_train = torch.randint(0, 2, (100, 1), dtype=torch.float32)
    X_val = torch.randn(20, 2, 32, 128, 128)
    y_val = torch.randint(0, 2, (20, 1), dtype=torch.float32)

    # 构建 Dataset 与 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数、优化器和学习率调度器的设置
    model = DenseNet3DTwoChannelFusion(num_classes=1, growth_rate=16).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 二分类常用的带 Sigmoid 的交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # -------------------------------
    # 训练函数定义
    # -------------------------------
    def train():
        model.train()  # 进入训练模式
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            # 使用 tqdm 显示训练进度
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # 计算预测准确率：先对输出进行 Sigmoid，再根据阈值判断类别
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            scheduler.step()  # 更新学习率
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}  Acc: {100 * correct / total:.2f}%")

    # -------------------------------
    # 验证函数定义
    # -------------------------------
    def evaluate():
        model.eval()  # 进入验证模式，不更新梯度
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                print(labels.size(0), (preds == labels).sum().item())
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # -------------------------------
    # 推理函数定义
    # -------------------------------
    def predict(image):
        """
        对单个样本进行推理
        image: numpy 数组，形状应为 (2, 32, 128, 128)
        """
        model.eval()
        image = torch.tensor(image).unsqueeze(0).to(device)  # 扩展 batch 维度
        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output).cpu().numpy()
        return prob

    # 运行训练与验证流程
    train()
    evaluate()

    # 示例推理
    import numpy as np
    test_sample = np.random.randn(2, 32, 128, 128).astype(np.float32)  # 生成一个随机测试样本，两通道数据
    prediction = predict(test_sample)
    print(f"Predicted Probability: {prediction}")
