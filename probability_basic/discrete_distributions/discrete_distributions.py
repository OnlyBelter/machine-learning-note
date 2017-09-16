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


def binom_dis(n=1, p=0.1):
    """
    二项分布，模拟抛硬币试验
    :param n: 实验总次数
    :param p: 单次实验成功的概率
    :return: 试验成功的次数
    """
    binom_dis = stats.binom(n, p)
    simulation_result = binom_dis.rvs(size=5)  # 取20个符合该分布的随机变量
    print(simulation_result)  # [ 7 11 13  8 13], 每次结果会不一样
    prob_10 = binom_dis.pmf(10)
    print(prob_10)  # 0.117



def poisson_dis(mu=3.0):
    """
    泊松分布
    :param mu: 单位时间（或单位面积）内随机事件的平均发生率
    :return:
    """
    mu = 2
    poisson_dist = stats.poisson(mu)
    X2 = np.arange(5)
    x_prob2 = poisson_dist.pmf(X2)
    plt.plot(X2, x_prob2)
    poisson_dist.pmf(3)  # 0.18, 恰好发生3次的概率


def compare_binom_poisson(mu=4, n1=8, n2=50):
    """
    二项分布与泊松分布的比较
    :param mu: 泊松分布的参数，保持mu不变
    :param n1: 第一个二项分布中的实验次数，n比较小
    :param n2: 第二个二项分布中的实验次数，n比较大
    :return:
    """
    # 为了具有可比性, 利用mu = n * p, 计算p
    p1 = mu/n1  # 二项分布中的参数，单次实验成功的概率
    p2 = mu/n2
    poisson_dist = stats.poisson(mu)  # 初始化泊松分布
    binom_dist1 = stats.binom(n1, p1)  # 初始化第一个二项分布
    binom_dist2 = stats.binom(n2, p2)  # 初始化第二个二项分布

    # 计算pmf
    X = np.arange(poisson_dist.ppf(0.0001), poisson_dist.ppf(0.9999))
    y_po = poisson_dist.pmf(X)
    print(X)
    print(y_po)
    y_bi1 = binom_dist1.pmf(X)
    y_bi2 = binom_dist2.pmf(X)

    # 作图
    # First group
    # 当n比较小，p比较大时，两者差别比较大
    plt.figure(1)
    plt.subplot(211)
    plt.plot(X, y_bi1, 'b-', label='binom1 (n={}, p={})'.format(n1, p1))
    plt.plot(X, y_po, 'r--', label='poisson (mu={})'.format(mu))
    plt.ylabel('Probability')
    plt.title('Comparing PMF of Poisson Dist. and Binomial Dist.')
    plt.legend(loc='best', frameon=False)

    # second group
    # 当n比较大，p比较小时，两者非常相似
    plt.subplot(212)
    plt.plot(X, y_bi2, 'b-', label='binom1 (n={}, p={})'.format(n2, p2))
    plt.plot(X, y_po, 'r--', label='poisson (mu={})'.format(mu))
    plt.ylabel('Probability')
    plt.legend(loc='best', frameon=False)
    plt.show()


if __name__ == '__main__':
    # bernoulli_distribution()
    # binom_dis(20, 0.5)
    compare_binom_poisson(mu=4, n1=8, n2=50)









