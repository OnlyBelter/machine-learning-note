import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


norm_dis = stats.norm(5, 3) # 利用相应的分布函数及参数，创建一个冻结的正态分布(frozen distribution)
x = np.linspace(-5, 15, 101)  # 在区间[-5, 15]上均匀的取101个点


# 计算该分布在x中个点的概率密度分布函数值(PDF)
pdf = norm_dis.pdf(x)

# 计算该分布在x中个点的累计分布函数值(CDF)
cdf = norm_dis.cdf(x)

# 下面是利用matplotlib画图
plt.figure(1)
# plot pdf
plt.subplot(211)  # 两行一列，第一个子图
plt.plot(x, pdf, 'b-',  label='pdf')
plt.ylabel('Probability')
plt.title(r'PDF/CDF of normal distribution')
plt.text(-5.0, .12, r'$\mu=5,\ \sigma=3$')  # 3是标准差，不是方差
plt.legend(loc='best', frameon=False)
# plot cdf
plt.subplot(212)
plt.plot(x, cdf, 'r-', label='cdf')
plt.ylabel('Probability')
plt.legend(loc='best', frameon=False)

plt.show()
