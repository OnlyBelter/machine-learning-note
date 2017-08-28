# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:47:10 2017

@author: xin
"""

# an example
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def example1():
    # 分布的参数初始化
    myDF = stats.norm(5, 3)  # Create the frozen distribution
    # 取101个等间距的x
    X = np.linspace(-5, 15, 101)
    # cdf, 累计分布函数
    y = myDF.cdf(X)  # Calculate the corresponding CDF
    plt.plot(X, y)


def bernoulli_distribution():
    # 伯努利分布
    # 只有一个参数：p，实验成功的概率
    p = 0.6
    bernoulli_dist = stats.bernoulli(p)

    # 伯努利分布的概率质量分布函数pmf
    p_heads = bernoulli_dist.pmf(1)  # 试验结果为1的概率, 规定为正面, 概率为0.6
    p_tails = bernoulli_dist.pmf(0)  # 试验结果为0的概率, 规定为反面, 概率为0.4

    # 取100个服从参数为0.6的伯努利分布的随机变量
    trials = bernoulli_dist.rvs(100)

    print(np.sum(trials))  # 63, 相当于1的个数

    # 100个随机变量的直方图, 相当于取出来的100个随机变量的概率质量分布
    plt.hist(trials/len(trials))
    # plt.show()
    plt.savefig('bernoulli_pmf.png', dpi=200)
    plt.close()

    # 0-2之间均匀的取100个点
    x = np.linspace(0, 2, 100)

    cdf = bernoulli_dist.cdf  # 相当于取出来的100个随机变量的累积分布函数(cdf)

    plt.plot(x, cdf(x))  # 上述伯努利分布在区间[0, 2]上的cdf图像
    # plt.show()
    plt.savefig('bernoulli_cdf.png', dpi=200)
    plt.close()


def other():
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

if __name__ == '__main__':
    bernoulli_distribution()









