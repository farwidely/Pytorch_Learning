# 混淆矩阵用于计算多分类问题的TP,FP,FN,TN
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = [2, 1, 0, 1, 2, 0]
y_pred = [2, 0, 0, 1, 2, 1]
C = confusion_matrix(y_true, y_pred)
print(C)

# 指定混淆矩阵的label
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# C = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])

# 1-混淆矩阵
confusion_matrix = np.array(
    [[9, 3, 2],
     [0, 6, 1],
     [1, 1, 7]])

# 2-TP/TN/FP/FN的计算
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
print(TP)
print(TN)
print(FP)
print(FN)

# 3-其他的性能参数的计算
TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate
TNR = TN / (TN + FP)  # Specificity/ true negative rate
PPV = TP / (TP + FP)  # Precision/ positive predictive value
NPV = TN / (TN + FN)  # Negative predictive value
FPR = FP / (FP + TN)  # Fall out/ false positive rate
FNR = FN / (TP + FN)  # False negative rate
FDR = FP / (TP + FP)  # False discovery rate
ACC = TP / (TP + FN)  # accuracy of each class
print(TPR)
print(TNR)
print(PPV)
print(NPV)
print(FPR)
print(FNR)
print(FDR)
print(ACC)

# copied from the two address in line54
# https://blog.csdn.net/SartinL/article/details/105844832 & https://blog.csdn.net/Hello_Chan/article/details/108672379
