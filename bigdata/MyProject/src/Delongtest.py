import pickle
from sklearn.metrics import roc_auc_score
from scipy import stats
import numpy as np 
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

# # DeLong测试比较两个AUC
# def delong_test(y_true, y_pred_proba1, y_pred_proba2):
#     """
#     使用DeLong测试比较两个预测模型的AUC
#     """
#     # 转换为NumPy数组
#     y_true = np.array(y_true)
#     y_1 = np.array(y_pred_proba1)
#     y_2 = np.array(y_pred_proba2)
    
#     # 计算正样本和负样本的索引
#     pos_idx = np.where(y_true == 1)[0]
#     neg_idx = np.where(y_true == 0)[0]
    
#     # 计算Mann-Whitney U统计量
#     n_pos = len(pos_idx)
#     n_neg = len(neg_idx)
    
#     # 计算第一个模型的AUC
#     auc_1 = auc(neg_idx, [y_1[i] for i in pos_idx])
    
#     # 计算第二个模型的AUC
#     auc_2 = auc(neg_idx, [y_2[i] for i in pos_idx])
    
#     # 计算AUC差异
#     delta_auc = auc_1 - auc_2
    
#     # 计算方差
#     var_auc_1 = (auc_1 * (1 - auc_1)) / (n_pos * n_neg)
#     var_auc_2 = (auc_2 * (1 - auc_2)) / (n_pos * n_neg)
    
#     # 计算协方差（简化版本）
#     cov_auc = 0.5 * (var_auc_1 + var_auc_2)
    
#     # 计算标准误差
#     se = np.sqrt(var_auc_1 + var_auc_2 - 2 * cov_auc)
    
#     # 计算z-score
#     z = delta_auc / se
    
#     # 计算p值（双侧检验）
#     p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
#     return {
#         'AUC_1': auc_1,
#         'AUC_2': auc_2,
#         'delta_AUC': delta_auc,
#         'p_value': p_value,
#         'z_score': z
#     }
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics

class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        self._show_result()

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p

    def _show_result(self):
        z,p=self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold :print("There is a significant difference")
        else:        print("There is NO significant difference")


# # Model A (random) vs. "good" model B
# preds_A = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# preds_B = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])
# actual=    np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
# DelongTest(preds_A,preds_B,actual)

# ============ 1. 读取模型1的pkl文件 ============
with open('/home/vipuser/Desktop/test_new_file/resnet18_20250404_192331/test_results_3/ensemble_probs.pkl', 'rb') as f:
    data1 = pickle.load(f)
probs1 = data1['ensemble_probs']
labels1 = data1['true_labels']

# ============ 2. 读取模型2的pkl文件 ============
with open('/home/vipuser/Desktop/Classification/RD_test_results/predictions/ExternalTest_ensemble_probs.pkl', 'rb') as f:
# with open('/home/vipuser/Desktop/Classification/test_results_radiomics_3/MLP_test_results.pkl', 'rb') as f:
    data2 = pickle.load(f)
probs2 = data2['ensemble_probs']
labels2 = data2['true_labels']

# ============ 3. 计算各自的AUC ============
# 如果是同一个测试集，这里一般要求 labels1 == labels2
auc1 = roc_auc_score(labels1, probs1)
auc2 = roc_auc_score(labels2, probs2)

print(f"Model1 AUC: {auc1:.4f}")
print(f"Model2 AUC: {auc2:.4f}")

# ============ 4. DeLong 检验比较AUC差异 ============
# 通常要求是同一批测试样本、同一组真实标签
# 也就是说 labels1 应该和 labels2 相同或基本一致
if np.array_equal(labels1, labels2):
    results = DelongTest(probs1, probs2, labels1)
else:
    print("两个文件的真实标签不一致，无法直接进行DeLong检验！")
