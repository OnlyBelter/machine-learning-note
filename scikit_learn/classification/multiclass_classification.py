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
from sklearn.model_selection import cross_val_score  # 用于训练集内部交叉验证
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler  # 用于输入数据的预处理
from sklearn.metrics import confusion_matrix  # 用于评价模型


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

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

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
# 建立模型，参数random_state用于shuffle data
sgd_clf = SGDClassifier(random_state=42)
# 训练模型，用所有的training data, 但是标签只有True(5)和False(非5)两类
sgd_clf.fit(X_train, y_train)
# 预测
predict_result = sgd_clf.predict([some_digit])
print(sgd_clf.classes_)  # 训练以后的模型就有这个值了，所有y值的集合
print(predict_result)
# print(X_train, y_train)

# 随机森林模型, 本身就是多类分类器
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
self_print('Random forest result')
print(forest_clf.predict([some_digit]))
print(forest_clf.predict_proba([some_digit]))
# print('training dataset cross validation...')
# print(cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy'))
# 对训练集做变换？？
self_print('scaling the inputs')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))

# 预测
y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
self_print('confusion matrix')
print(conf_mx)
# plt.matshow(conf_mx)
# plt.show()
#--- plot normalized confusion matrix
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

# 查看分错类的样本
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8, 8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
