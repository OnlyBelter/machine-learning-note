# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from public_function import self_print
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent

# MINIST dataset
# mnist = fetch_mldata('MNIST original')
custom_data_home = r'D:\github\datasets'
# URL, http://mldata.org/repository/data/download/matlab/mnist-original/
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
X, y = mnist['data'], mnist['target']
self_print('X shape & y shape')
print(X.shape)  # (70000, 784), 28*28 pixels
print(y.shape)
# print(mnist)

some_digit = X[36000]

# show digit's image
def show_digit_image(digit_features):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()
# show_digit_image(some_digit)
# print(y[36000])

# The MNIST dataset is acturally already split into a training set(the first 60,000 images)
# and a test set(the last 10,000 images)
train_num = 60000
X_train, X_test, y_train, y_test = X[:train_num], X[train_num:], y[:train_num], y[train_num:]

# shuffle the training set, 打乱训练样本的次序，这一步在有些时候挺有用的
# 比如样本是排过序的，shuffle后可以保证每种数字的分布是均匀的
shuffle_index = np.random.permutation(60000) # 这个函数不错
# print(y_train)
# print(shuffle_index)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

## training a binary classifier
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)

# 建立模型，参数random_state用于shuffle data
sgd_clf = SGDClassifier(random_state=42)
# 训练模型，用所有的training data, 但是标签只有True(5)和False(非5)两类
sgd_clf.fit(X_train, y_train_5)
# 预测
predict_result = sgd_clf.predict([some_digit])
print(predict_result)

## performance measures
# measuring accuracy using cross-validation
# 利用训练样本进行交叉验证，从训练样本中划分出一小部分作为测试样本
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone  # deep copy model without data
def self_cross_val_score(sgd_clf, X, y, cv):
    skfolds = StratifiedKFold(n_splits=cv, random_state=42)  # 按照不同的方式取三次样，三次训练样本的个数相同
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)  # 对预测正确的数字计数
        return (n_correct / len(y_pred))

# 上面的方法等价于下面的cross_val_score
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
self_print('cross validation score')
print(cv_score)

## confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# 还是不太明白这里的cross_val_predict是如何工作的
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(y_train_pred.shape)
con_matrix = confusion_matrix(y_train_5, y_train_pred)
self_print('confusion matrix')
print(con_matrix)
precision_sco =  precision_score(y_train_5, y_train_pred)
recall_sco = recall_score(y_train_5, y_train_pred)
self_print('precision & recall')
print(precision_sco, recall_sco)
f1_score(y_train_5, y_train_pred)  # F1 score is the harmoinc mean of precision and recall

# 所有y的分数(用于与阈值作比较)，不同的阈值导致最后分类的效果不同
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
self_print('y_scores:')
print(y_scores, y_scores.shape)
# 利用函数precision_recall_curve计算不同阈值下预测的准确率和召回率
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print(precisions)
print(recalls)
print(thresholds)
self_print('plot precision recall vs threshold')
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

self_print('plot ROC curve')
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')  # 1 - specificity or 1 - precision
    plt.ylabel('True Positive Rate')  # recall
# plot_roc_curve(fpr, tpr)
# plt.show()

# 计算AUC(area under the curve), 最好的分类器AUC=1, 随机分类器AUC=0.5
print('AUC is', str(roc_auc_score(y_train_5, y_scores)))


## random forest classifier, 与随机梯度下降分类器作比较
# RandomForestClassifier class 没有decision_function()方法
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
# print(y_probas_forest, y_probas_forest.shape)
# print(X_train[36000])
# print(y_probas_forest[36000])
y_scores_forest = y_probas_forest[:, 1]  # score = probability of positive class
print(y_scores_forest)
#---- compare ROC
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()
print('Random forest\'s AUC is', (roc_auc_score(y_train_5, y_scores_forest)))

