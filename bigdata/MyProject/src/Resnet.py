import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False), 
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        num_seg_classes,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False) 
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model



class BinaryResNet3D_ParallelBranch(nn.Module):
    def __init__(self, model_depth=50, num_input_channels=1):
        super(BinaryResNet3D_ParallelBranch, self).__init__()
        
        # 确定使用哪种block和层数配置
        if model_depth == 10:
            block = BasicBlock
            layers = [1, 1, 1, 1]
        elif model_depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif model_depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif model_depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif model_depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        
        # 创建MRI分支（使用完整的ResNet前半部分）
        self.mri_inplanes = 64
        self.mri_conv1 = nn.Conv3d(
            num_input_channels, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.mri_bn1 = nn.BatchNorm3d(64)
        self.mri_relu = nn.ReLU(inplace=True)
        self.mri_maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.mri_layer1 = self._make_layer(block, 64, layers[0], branch='mri')
        self.mri_layer2 = self._make_layer(block, 128, layers[1], stride=2, branch='mri')
        
        # 创建掩码分支（独立参数，但相同结构）
        self.mask_inplanes = 64
        self.mask_conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.mask_bn1 = nn.BatchNorm3d(64)
        self.mask_relu = nn.ReLU(inplace=True)
        self.mask_maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.mask_layer1 = self._make_layer(block, 64, layers[0], branch='mask')
        self.mask_layer2 = self._make_layer(block, 128, layers[1], stride=2, branch='mask')
        
        # 共享尾部（layer3和layer4）
        self.shared_inplanes = 128 * block.expansion
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, branch='shared')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, branch='shared')
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 最终分类器
        if block == BasicBlock:
            fc_in_features = 512  # BasicBlock的expansion=1
        else:  # Bottleneck
            fc_in_features = 512 * 4  # Bottleneck的expansion=4
        
        self.fc = nn.Linear(fc_in_features, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, branch='shared'):
        """
        创建ResNet的layer，为不同分支使用不同的self.inplanes
        
        Args:
            block (nn.Module): BasicBlock或Bottleneck
            planes (int): 该层的基本通道数
            blocks (int): 该层包含的block数量
            stride (int): 第一个block的步长
            dilation (int): 卷积的膨胀率
            branch (str): 分支名称('mri', 'mask', 或'shared')
            
        Returns:
            nn.Sequential: 包含多个block的层
        """
        downsample = None
        
        # 根据分支选择正确的inplanes
        if branch == 'mri':
            inplanes = self.mri_inplanes
        elif branch == 'mask':
            inplanes = self.mask_inplanes
        else:  # 'shared'
            inplanes = self.shared_inplanes
        
        # 当步长不为1或输入通道与输出通道不匹配时，需要下采样
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
            
        layers = []
        
        # 添加第一个block（可能有下采样和不同的步长）
        layers.append(block(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        
        # 重要！更新相应分支的inplanes值为扩展后的通道数
        if branch == 'mri':
            self.mri_inplanes = planes * block.expansion
        elif branch == 'mask':
            self.mask_inplanes = planes * block.expansion
        else:  # 'shared'
            self.shared_inplanes = planes * block.expansion
        
        # 添加剩余的block，它们的输入通道数应为planes * block.expansion
        for i in range(1, blocks):
            # 注意：这里使用更新后的通道数作为输入，而不是原始的inplanes
            if branch == 'mri':
                layers.append(block(self.mri_inplanes, planes, dilation=dilation))
            elif branch == 'mask':
                layers.append(block(self.mask_inplanes, planes, dilation=dilation))
            else:  # 'shared'
                layers.append(block(self.shared_inplanes, planes, dilation=dilation))
                
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def load_pretrained_weights(self, pretrained_path, init_mask_from_mri=True):
        """
        从预训练的ResNet模型加载权重到双分支网络
        
        Args:
            pretrained_path: 预训练ResNet模型的路径
            init_mask_from_mri: 是否用MRI分支的权重初始化掩码分支(可选)
        """
        print(f"从 {pretrained_path} 加载预训练权重...")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # 处理可能的不同保存格式
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        
        # 移除可能的'module.'前缀(来自DataParallel)
        if list(pretrained_dict.keys())[0].startswith('module.'):
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        
        # 当前模型参数
        model_dict = self.state_dict()
        
        # 存储将加载的权重
        new_dict = {}
        skipped_layers = []
        
        # 1. 映射预训练权重到MRI分支和共享尾部
        for k, v in pretrained_dict.items():
            # 跳过不需要的层(分割头或分类器)
            if k.startswith('fc.') or k.startswith('conv_seg.'):
                continue
            
            # 映射到MRI分支
            if k.startswith('conv1.') or k.startswith('bn1.'):
                new_key = 'mri_' + k
                if new_key in model_dict and model_dict[new_key].shape == v.shape:
                    new_dict[new_key] = v
                else:
                    skipped_layers.append((new_key, k))
            
            # 映射layer1、layer2到MRI分支
            elif k.startswith('layer1.') or k.startswith('layer2.'):
                new_key = 'mri_' + k
                if new_key in model_dict and model_dict[new_key].shape == v.shape:
                    new_dict[new_key] = v
                else:
                    skipped_layers.append((new_key, k))
            
            # 映射layer3、layer4到共享尾部
            elif k.startswith('layer3.') or k.startswith('layer4.'):
                if k in model_dict and model_dict[k].shape == v.shape:
                    new_dict[k] = v
                else:
                    skipped_layers.append((k, k))
        
        # 2. 可选：用MRI分支权重初始化掩码分支
        if init_mask_from_mri:
            # 创建临时字典存储掩码分支参数
            mask_dict = {}
            for k, v in new_dict.items():
                if k.startswith('mri_'):
                    mask_key = k.replace('mri_', 'mask_')
                    if mask_key in model_dict and model_dict[mask_key].shape == v.shape:
                        mask_dict[mask_key] = v.clone()
            # 迭代完成后再更新原字典
            new_dict.update(mask_dict)
        # 3. 更新模型权重
        model_dict.update(new_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"成功加载 {len(new_dict)} 个预训练参数")
        if skipped_layers:
            print(f"跳过 {len(skipped_layers)} 个不匹配的层")

    def forward(self, x):
        """
        x shape: (B, 2, D, H, W)
          where x[:, 0, ...] = MRI, x[:, 1, ...] = Mask
        """
        # 分离输入数据
        x_mri = x[:, 0:1, ...]    # shape (B, 1, D, H, W)
        x_mask = x[:, 1:2, ...]   # shape (B, 1, D, H, W)
        
        # MRI分支前向传播
        x_mri = self.mri_conv1(x_mri)
        x_mri = self.mri_bn1(x_mri)
        x_mri = self.mri_relu(x_mri)
        x_mri = self.mri_maxpool(x_mri)
        x_mri = self.mri_layer1(x_mri)
        x_mri = self.mri_layer2(x_mri)
        
        # 掩码分支前向传播
        x_mask = self.mask_conv1(x_mask)
        x_mask = self.mask_bn1(x_mask)
        x_mask = self.mask_relu(x_mask)
        x_mask = self.mask_maxpool(x_mask)
        x_mask = self.mask_layer1(x_mask)
        x_mask = self.mask_layer2(x_mask)
        
        # 融合特征（确保形状相同）
        x_fused = x_mri + x_mask
        
        # 共享尾部前向传播
        x_fused = self.layer3(x_fused)
        x_fused = self.layer4(x_fused)
        
        # 全局平均池化
        x_fused = self.avgpool(x_fused)
        x_fused = x_fused.view(x_fused.size(0), -1)
        
        # 分类器
        logit = self.fc(x_fused)
        
        return logit


class BinaryResNet3D_SingleBranch(nn.Module):
    def __init__(self, model_depth=50, num_input_channels=1):
        super(BinaryResNet3D_SingleBranch, self).__init__()
        
        # 确定使用哪种block和层数配置
        if model_depth == 10:
            block = BasicBlock
            layers = [1, 1, 1, 1]
        elif model_depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif model_depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif model_depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif model_depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        
        # 创建MRI分支（现在是主分支）
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            num_input_channels, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 最终分类器
        if block == BasicBlock:
            fc_in_features = 512  # BasicBlock的expansion=1
        else:  # Bottleneck
            fc_in_features = 512 * 4  # Bottleneck的expansion=4
        
        self.fc = nn.Linear(fc_in_features, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """创建ResNet的layer"""
        downsample = None
        
        # 当步长不为1或输入通道与输出通道不匹配时，需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
            
        layers = []
        
        # 添加第一个block（可能有下采样和不同的步长）
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        
        # 更新inplanes值为扩展后的通道数
        self.inplanes = planes * block.expansion
        
        # 添加剩余的block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
                
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def load_pretrained_weights(self, pretrained_path):
        """
        从预训练的ResNet模型加载权重
        
        Args:
            pretrained_path: 预训练ResNet模型的路径
        """
        print(f"从 {pretrained_path} 加载预训练权重...")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # 处理可能的不同保存格式
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        
        # 移除可能的'module.'前缀(来自DataParallel)
        if list(pretrained_dict.keys())[0].startswith('module.'):
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        
        # 当前模型参数
        model_dict = self.state_dict()
        
        # 存储将加载的权重
        new_dict = {}
        skipped_layers = []
        
        # 映射预训练权重到MRI分支和共享尾部
        for k, v in pretrained_dict.items():
            # 跳过不需要的层(分割头或分类器)
            if k.startswith('fc.') or k.startswith('conv_seg.'):
                continue
            
            # 映射到主干网络
            if k in model_dict and model_dict[k].shape == v.shape:
                new_dict[k] = v
            else:
                skipped_layers.append(k)
        
        # 更新模型权重
        model_dict.update(new_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"成功加载 {len(new_dict)} 个预训练参数")
        if skipped_layers:
            print(f"跳过 {len(skipped_layers)} 个不匹配的层")

    def forward(self, x):
        """
        x shape: (B, 2, D, H, W)
        仍接受双通道输入，但只使用第一通道(MRI)
        """
        # 只提取MRI通道
        x_mri = x[:, 0:1, ...]    # shape (B, 1, D, H, W)
        
        # 前向传播
        x = self.conv1(x_mri)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类器
        logit = self.fc(x)
        
        return logit
    

    
class BinaryResNet3D_FusionModel(nn.Module):
    def __init__(self, model_depth=50, num_input_channels=1, freeze_branches=True):
        super(BinaryResNet3D_FusionModel, self).__init__()
        
        # Create the dual-branch model
        self.local_branch_model = BinaryResNet3D_ParallelBranch(
            model_depth=model_depth, 
            num_input_channels=num_input_channels
        )
        
        # Create the single-branch model
        self.global_branch_model = BinaryResNet3D_ParallelBranch(
            model_depth=model_depth, 
            num_input_channels=num_input_channels
        )
        
        # Determine the feature dimension based on the block type
        if model_depth in [10, 18, 34]:  # BasicBlock models
            feat_dim = 512
        else:  # Bottleneck models (50, 101)
            feat_dim = 512 * 4  # Bottleneck expansion = 4
            
        # Optionally freeze the branch models
        if freeze_branches:
            for param in self.local_branch_model.parameters():
                param.requires_grad = False
            for param in self.global_branch_model.parameters():
                param.requires_grad = False
        
        # Define fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim * 2),  # Add normalization to standardize features
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Lower dropout rate
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        # self.fusion_classifier = nn.Sequential(
        #     nn.BatchNorm1d(feat_dim * 2),  # Add normalization to standardize features
        #     nn.Linear(feat_dim * 2, 256),
        #     nn.BatchNorm1d(256),  # Add normalization to standardize features
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),  # Lower dropout rate
        #     nn.Linear(256, 32),
        #     nn.BatchNorm1d(32),  # Add normalization to standardize features
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),  # Lower dropout rate
        #     nn.Linear(32, 1)
        # )

        # Add attention mechanisms
        self.attention_dual = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
            nn.Sigmoid()
        )
        
        self.attention_single = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
            nn.Sigmoid()
        )
        
    def forward(self, g_in, l_in):
        """
        Forward pass for the fusion model
        
        Returns:
            Tensor of shape (B, 1) with the classification logits
        """
        # Extract features
        dual_features = self._forward_features_local(l_in)
        single_features = self._forward_features_global(g_in)
        
        # Calculate attention weights for each feature set
        attn_dual = self.attention_dual(dual_features)
        attn_single = self.attention_single(single_features)
        
        # Apply attention and combine
        weighted_dual = dual_features * attn_dual
        weighted_single = single_features * attn_single
        
        # Concatenate weighted features
        concat_features = torch.cat([weighted_dual, weighted_single], dim=1)
        
        # Apply fusion classifier
        logits = self.fusion_classifier(concat_features)
        
        return logits
    
    def _forward_features_local(self, x):
        """Extract features from dual-branch model (without final classification)"""
        # Split input data
        x_mri = x[:, 0:1, ...]
        x_mask = x[:, 1:2, ...]
        
        # MRI branch forward pass
        x_mri = self.local_branch_model.mri_conv1(x_mri)
        x_mri = self.local_branch_model.mri_bn1(x_mri)
        x_mri = self.local_branch_model.mri_relu(x_mri)
        x_mri = self.local_branch_model.mri_maxpool(x_mri)
        x_mri = self.local_branch_model.mri_layer1(x_mri)
        x_mri = self.local_branch_model.mri_layer2(x_mri)
        
        # Mask branch forward pass
        x_mask = self.local_branch_model.mask_conv1(x_mask)
        x_mask = self.local_branch_model.mask_bn1(x_mask)
        x_mask = self.local_branch_model.mask_relu(x_mask)
        x_mask = self.local_branch_model.mask_maxpool(x_mask)
        x_mask = self.local_branch_model.mask_layer1(x_mask)
        x_mask = self.local_branch_model.mask_layer2(x_mask)
        
        # Fusion and shared layers
        x_fused = x_mri + x_mask
        x_fused = self.local_branch_model.layer3(x_fused)
        x_fused = self.local_branch_model.layer4(x_fused)
        
        # Global average pooling
        x_fused = self.local_branch_model.avgpool(x_fused)
        features = x_fused.view(x_fused.size(0), -1)
        
        return features


    def _forward_features_global(self, x):
        """Extract features from dual-branch model (without final classification)"""
        # Split input data
        x_mri = x[:, 0:1, ...]
        x_mask = x[:, 1:2, ...]
        
        # MRI branch forward pass
        x_mri = self.global_branch_model.mri_conv1(x_mri)
        x_mri = self.global_branch_model.mri_bn1(x_mri)
        x_mri = self.global_branch_model.mri_relu(x_mri)
        x_mri = self.global_branch_model.mri_maxpool(x_mri)
        x_mri = self.global_branch_model.mri_layer1(x_mri)
        x_mri = self.global_branch_model.mri_layer2(x_mri)
        
        # Mask branch forward pass
        x_mask = self.global_branch_model.mask_conv1(x_mask)
        x_mask = self.global_branch_model.mask_bn1(x_mask)
        x_mask = self.global_branch_model.mask_relu(x_mask)
        x_mask = self.global_branch_model.mask_maxpool(x_mask)
        x_mask = self.global_branch_model.mask_layer1(x_mask)
        x_mask = self.global_branch_model.mask_layer2(x_mask)
        
        # Fusion and shared layers
        x_fused = x_mri + x_mask
        x_fused = self.global_branch_model.layer3(x_fused)
        x_fused = self.global_branch_model.layer4(x_fused)
        
        # Global average pooling
        x_fused = self.global_branch_model.avgpool(x_fused)
        features = x_fused.view(x_fused.size(0), -1)
        
        return features

    def load_pretrained_weights(self, local_path=None, global_path=None, device='cpu'):
        """
        Load pretrained weights for both branches
        
        Args:
            local_path: Path to pretrained weights for dual-branch model
            global_path: Path to pretrained weights for single-branch model
            device: Device to load the weights on
        """
        if local_path:
            local_state_dict = torch.load(local_path, map_location=device)
            if 'model_state_dict' in local_state_dict:
                local_state_dict = local_state_dict['model_state_dict']
            self.local_branch_model.load_state_dict(local_state_dict, strict=False)
            print(f"Loaded dual-branch weights from {local_path}")
            
        if global_path:
            global_state_dict = torch.load(global_path, map_location=device)
            if 'model_state_dict' in global_state_dict:
                global_state_dict = global_state_dict['model_state_dict']
            self.global_branch_model.load_state_dict(global_state_dict, strict=False)
            print(f"Loaded single-branch weights from {global_path}")



# 在文件末尾添加以下测试代码

# ResNet模型的形状打印器（使用hooks）
class LayerShapePrinter:
    def __init__(self, model):
        self.model = model
        self.handles = []
        
    def register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                print(f"{name} 输出形状: {output.shape}")
            return hook
        
        # 为重要层注册hooks
        self.handles.append(self.model.conv1.register_forward_hook(hook_fn("conv1")))
        self.handles.append(self.model.bn1.register_forward_hook(hook_fn("bn1")))
        self.handles.append(self.model.layer1.register_forward_hook(hook_fn("layer1")))
        self.handles.append(self.model.layer2.register_forward_hook(hook_fn("layer2")))
        self.handles.append(self.model.layer3.register_forward_hook(hook_fn("layer3")))
        self.handles.append(self.model.layer4.register_forward_hook(hook_fn("layer4")))
        if hasattr(self.model, 'conv_seg'):
            self.handles.append(self.model.conv_seg.register_forward_hook(hook_fn("conv_seg")))
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

# —— 直接添加在文件末尾 ——  

if __name__ == "__main__":
    # 设备配置
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # —— 创建模型（以 ResNet18 为例，按需改成 resnet50 等） ——  
    sample_input_D, sample_input_H, sample_input_W = 37, 185, 250
    num_seg_classes = 2  # 如果不做分割，可随意填
    model = resnet50(
        sample_input_D=sample_input_D,
        sample_input_H=sample_input_H,
        sample_input_W=sample_input_W,
        num_seg_classes=num_seg_classes
    ).to(device)
    
    # 引入 hook 打印工具
    shape_printer = LayerShapePrinter(model)
    shape_printer.register_hooks()
    
    # 构造自定义尺寸输入并前向
    batch_size = 1
    custom_input = torch.randn(batch_size, 1, sample_input_D, sample_input_H, sample_input_W).to(device)
    with torch.no_grad():
        out = model(custom_input)
    
    print(f"\n模型对 (1, 1, {sample_input_D}, {sample_input_H}, {sample_input_W}) 输入的最终输出形状: {out.shape}")
    
    # 清理 hooks
    shape_printer.remove_hooks()
