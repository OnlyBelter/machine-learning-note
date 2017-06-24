# -*- coding: utf-8 -*-
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
print(y_train)
print(shuffle_index)
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
skfolds = StratifiedKFold(n_splits=3, random_state=42)  #




