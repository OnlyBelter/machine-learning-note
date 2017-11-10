import numpy as np

# 参考
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html


data = np.arange(7, 14)
print(data)  # [ 7  8  9 10 11 12 13]

## 计算方差
# 直接使用样本二阶中心距计算方差，分母为n
var_n = np.var(data)  # 默认，ddof=0
print(var_n) # 4.0
# 使用总体方差的无偏估计计算方差，分母为n-1
var_n_1 = np.var(data, ddof=1)  # 使用ddof设置自由度的偏移量
print(var_n_1) # 4.67


## 计算标准差
std_n = np.std(data, ddof=0)
std_n_minus_1 = np.std(data, ddof=1)  # 使用ddof设置自由度的偏移量
print(std_n, std_n_minus_1)  # 2.0, 2.16
print(std_n**2, std_n_minus_1**2)  # 4.0, 4.67

