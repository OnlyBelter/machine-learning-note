# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:47:10 2017

@author: xin
"""

# an example
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# 分布的参数初始化
myDF = stats.norm(5,3) # Create the frozen distribution
# 取101个等间距的x
X = np.linspace(-5, 15, 101)
# cdf, 累计分布函数
y = myDF.cdf(X) # Calculate the corresponding CDF
plt.plot(X, y)


# 伯努利分布
# 只有一个参数：p，实验成功的概率
from scipy import stats
p = 0.6
bernoulliDist = stats.bernoulli(p)
p_tails = bernoulliDist.pmf(0)
p_heads = bernoulliDist.pmf(1)

trials = bernoulliDist.rvs(10)

# 二项分布
# 两个参数：n,p (实验总次数，单次实验成功的概率)
from scipy import  stats
import numpy as np
import matplotlib.pyplot as plt
(p, n) = (0.5, 4)  # 四次实验，每次实验成功的概率为0.5
binomDist = stats.binom(n, p)
X = np.arange(5)
x_prob = binomDist.pmf(X)
plt.plot(X, x_prob)
binomDist.pmf(3)  # 0.25, 在4次实验中恰好成功3次的概率


# 泊松分布
# 泊松分布只有一个参数：mu, 表示单位时间（或单位面积）内随机事件的平均发生率
from scipy import  stats
import numpy as np
import matplotlib.pyplot as plt
mu = 2
poissonDist = stats.poisson(mu)
X2 = np.arange(5)
x_prob2 = poissonDist.pmf(X2)
plt.plot(X2, x_prob2)
poissonDist.pmf(3)  # 0.18, 恰好发生3次的概率

# 二项分布与泊松分布的比较
from scipy import  stats
import numpy as np
import matplotlib.pyplot as plt
mu = 4  # 保持mu不变，泊松分布的参数
n1 = 8  # 二项分布中的实验次数
n2 = 50
p1 = mu/n1  # 二项分布中的参数，单次实验成功的概率
p2 = mu/n2  # mu = n * p
X = np.arange(9)
binomDist1 = stats.binom(n1, p1)
binomDist2 = stats.binom(n2, p2)
poissonDist = stats.poisson(mu)
y_bi1 = binomDist1.pmf(X)
y_bi2 = binomDist2.pmf(X)
y_po = poissonDist.pmf(X)
# First group
# 当n比较小，p比较大时，两者差别比较大
plt.plot(X, y_bi1)
plt.plot(X, y_po)


# second group
# 当n比较大，p比较小时，两者非常相似
plt.plot(X, y_bi2)
plt.plot(X, y_po)


