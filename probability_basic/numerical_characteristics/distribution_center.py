import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## 计算平均值
x = np.arange(1, 11)
print(x)  # [ 1  2  3  4  5  6  7  8  9 10]
mean = np.mean(x)
print(mean)  # 5.5

# 对空值的处理，nan stands for 'Not-A-Number'
x_with_nan = np.hstack((x, np.nan))
print(x_with_nan)  # [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  nan]
mean2 = np.mean(x_with_nan)
print(mean2)  # nan，直接计算没有结果
mean3 = np.nanmean(x_with_nan)
print(mean3)  # 5.5

## 计算几何平均值
x2 = np.arange(1, 11)
print(x2)  # [ 1  2  3  4  5  6  7  8  9 10]
geometric_mean = stats.gmean(x2)
print(geometric_mean)  # 4.52872868812，几何平均值小于等于算数平均值
