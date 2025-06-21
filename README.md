# dissertation_code

test61.py <--> Task02_PASp61：部分预处理后的训练集和内部测试集。
test16.py <--> Task02_PASp61/labelsTs：nnUNet分割得到的内部测试集的分割结果掩码。

test14.py <--> Task02_PASp62_edge：将原本标签里的ROI替换为对应ROI的边缘后的，部分预处理后的训练集和内部测试集。

test15.py <--> Task02_PASp62：全部预处理后的训练集和内部测试集。

test18.py <--> Task02_PASp62_radiomics：只是集中于标签部分的，用于提取影像组学特征的训练集和内部测试集。
test18.py <--> train_features.csv, test_features.csv：训练集和测试集对应的影像组学特征
test19.py <--> train_labels.csv, test_labels.csv：训练集和测试集对应的标签



extract              Center3/extract
create              Center3/create
selection           Center3/selection
train
test                    test_3



extract_RD            Center3/extract_RD

selection_RD           Center3/selection_RD
train_RD 
test_RD                    test_RD_3


实验错误：
1. 从5个网络里选择最好的那个网络，我用的是随机种子配对，AUC相加，然后选取中位数的方法。但是这样不一定正确（对应每个模型下的ensemble_models文件夹）。
2. 全局分支和局部分支，全连接层应该是512-128-1。现在是512-1。
3. 双分支网络最后的分类头，应该是我注释的那种写法。现在写的太简单了。
4. 在dataset里加载center3的数据时，应该用和训练集一样的final_shape。但现在用的是根据center3数据计算得到的专门的final_shape。
5. dropout丢弃率应该是0.3。但现在影像组学的是0.1，是为了让MLP表现差一点。