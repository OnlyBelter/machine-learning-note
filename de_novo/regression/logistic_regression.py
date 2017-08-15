'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import os

path = r'..\data'
training_sample = 'Logistic_Regression-trainingSample.txt'
testing_sample = 'Logistic_Regression-testingSample.txt'

# 从文件中读入训练样本的数据
def loadDataSet(p, file_n):
    dataMat = []
    labelMat = []
    fr = open(os.path.join(p, file_n))
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 三个特征x0, x1, x2, x0=1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))  # 样本标签y
    return dataMat, labelMat

def sigmoid(X):
    return 1.0/(1+exp(-X))

# 梯度下降法求回归系数a
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             # 转换成numpy中的矩阵, X, 90 x 3
    labelMat = mat(classLabels).transpose()  # 转换成numpy中的矩阵, y, 90 x 1
    m, n = shape(dataMatrix)  # m=90, n=3
    alpha = 0.001  # 学习率
    maxCycles = 1000
    weights = ones((n, 1))  # 初始参数, 3 x 1
    for k in range(maxCycles):              # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)     # 模型预测值, 90 x 1
        error = h - labelMat              # 真实值与预测值之间的误差, 90 x 1
        temp = dataMatrix.transpose() * error  # 所有参数的偏导数, 3 x 1
        weights = weights - alpha * temp  # 更新权重
    return weights

# 测试函数
def test_logistic_regression():
    dataArr, labelMat = loadDataSet(path, training_sample)  # 读入训练样本中的原始数据
    A = gradAscent(dataArr, labelMat)  # 回归系数a的值
    h = sigmoid(mat(dataArr)*A)  # 预测结果h(a)的值
    print(dataArr, labelMat)
    print(A)
    print(h)
    # plotBestFit(A.getA())

test_logistic_regression()
