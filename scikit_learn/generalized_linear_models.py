# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:26:08 2017

@author: Belter
"""

# 广义线性模型
# 基本假设：目标值是由输入变量的线性组合得到的
# http://scikit-learn.org/stable/modules/linear_model.html

#------------------------ part1: 系数由普通最小二乘法估计得到--------------------
# http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# 基本假设：变量之间是相互独立的
from sklearn import linear_model
reg = linear_model.LinearRegression()
X = [[0, 0], [1, 1], [2, 2]] # Training data, 形状：[n_samples, n_features]
y = [0, 1, 2]  # Target values,  形状：[n_samples, n_targets]
reg.fit(X, y)  # 训练
# 由下面两个参数可以得到拟合的直线方程为 y = 0.5*x_1 + 0.5*x_2
reg.coef_  # 系数的值， array([ 0.5,  0.5])
reg.intercept_  #  截距， 2.2204460492503131e-16
reg.predict([8, 9])  # 预测一个新的x的值，array([ 8.5])


#---------- 一个更复杂的例子
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# 数据集中一共有10个特征，这里只取其中的一个
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# 拟合的直线方程的系数
print('Coefficients: \n', regr.coef_)  # 938.23786125
# 计算测试样本中的均方误差(The mean squared error, MSE)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction, 可释方差，越接近1越好
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
# plot所有的测试数据
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# 根据测试数据中的x,使用拟合的直线方尺预测y
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)


plt.show()


#------------------------ part2: 系数由岭回归估计得到--------------------
# http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
# 基本假设：变量之间是有相互依赖的，多重共线性问题
# 添加了一个惩罚项进行正则化，对X进行了奇异值分解
# This method computes the least squares solution using a singular value decomposition of X
# This example also shows the usefulness of applying Ridge regression to highly ill-conditioned matrices. 
# For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. 
# In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).
# 在存在线性依赖的情况下，y的轻微变化就会引起参数的剧烈波动

from sklearn import linear_model
# alpha相当于正则化系数，该值非常大的时候，参数会趋近于0；该值等于0时，模型与普通最小二乘法相同
# 加这个参数的目的是，降低参数对y的敏感性
reg = linear_model.Ridge(alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
reg.coef_  # array([ 0.34545455,  0.34545455])
reg.intercept_  # 0.13636363636363641


#---------- alpha与系数的关系
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py
# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)  # X有10个特征，因此有10个参数
    coefs.append(clf.coef_)

# alphas=1.00000000e-10, 系数之间差异很大
# coef = array([   2.64506194,  -27.6037085 ,    7.99288375,  133.67548533, 18.04324303, 
#               -123.85506905, -175.62005751, -113.78633294, 45.15379189,  274.02303879])
# alphas=1.00000000e-02, 系数之间差异很小
# coef = array([-1.15365551, -0.06380733,  0.82265094,  1.33384561,  1.62104261,
#             1.77805326,  1.85752347,  1.88963634,  1.89230434,  1.87650476])
    
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


#---------- 使用交叉验证的方法选择正则化系数
# generalized Cross-Validation
# RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter. 
# The object works in the same way as GridSearchCV except that it defaults to Generalized Cross-Validation (GCV), an efficient form of leave-one-out cross-validation:
# 没有交叉验证数据时，可以采用leave-one-out cross-validation
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       

reg.alpha_  # 0.1
reg.coef_  # array([ 0.44186047,  0.44186047])
reg.intercept_  # 0.072093023255812183

