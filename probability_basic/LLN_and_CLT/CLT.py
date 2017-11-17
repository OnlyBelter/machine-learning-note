# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Created on Sun Nov 12 08:44:37 2017

@author: Belter
"""


def sampling2pmf(n, dist, m=100000):
    """
    n: sample size for each experiment
    m: how many times do you do experiment, fix in 100000
    dist: frozen distribution
    """
    current_dist = dist
    sum_of_samples = []
    for i in range(m):
        samples = current_dist.rvs(size=n)  # 与每次取一个值，取n次效果相同
        # print(samples)
        sum_of_samples.append(np.sum(samples))
    # 下面计算频率的方式不适合连续型随机变量，因此直接返回随机变量的和
    # val, cnt = np.unique(sum_of_samples, return_counts=True)
    # pmf = cnt / len(sum_of_samples)
    # return val, pmf
    return sum_of_samples


def plot(n, dist, subplot, plt_handle, dist_type):
    """
    :param n: sample size
    :param dist: distribution of each single sample
    :param subplot: location of sub-graph, such as 221, 222, 223, 224
    :param plt_handle: plt object
    :param dist_type: the type of distribution
    :return: plt object
    """
    bins = 20000
    plt = plt_handle
    plt.subplot(subplot)
    mu = n * dist.mean()
    sigma = np.sqrt(n * dist.var())
    samples = sampling2pmf(n=n, dist=dist)
    # print(samples)
    # normed参数可以对直方图进行标准化，从而使纵坐标表示概率而不是次数
    plt.hist(samples, normed=True, bins=50, histtype='stepfilled', alpha=1)
    plt.ylabel('Probability')
    plt.title('Sum of {} dist. (n={})'.format(dist_type, n))
    # normal distribution
    norm_dis = stats.norm(mu, sigma)
    norm_x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, bins)
    pdf1 = norm_dis.pdf(norm_x)
    plt.plot(norm_x, pdf1, 'r--', alpha=0.4)
    return plt

size = [1, 2, 3, 4, 8, 10]

# sum of bernoulli distribution
# dist_type = 'bern'
# bern_para = [0.99]
# single_sample_dist = stats.bernoulli(p=bern_para[0])  # 定义一个伯努利分布

# sum of binomial distribution
# dist_type = 'bino'
# bino_para = [20, 0.4]
# single_sample_dist = stats.binom(n=bino_para[0], p=bino_para[1])  # 定义一个二项分布

# sum of uniform distribution
dist_type = 'uniform'
uniform_para = [3, 2]
single_sample_dist = stats.uniform(loc=uniform_para[0], scale=uniform_para[1])  # 定义一个均匀分布


# 下面是利用matplotlib画图
plt.figure(1)
plt = plot(n=size[0], dist=single_sample_dist, subplot=321, plt_handle=plt, dist_type=dist_type)
plt = plot(n=size[1], dist=single_sample_dist, subplot=322, plt_handle=plt, dist_type=dist_type)
plt = plot(n=size[2], dist=single_sample_dist, subplot=323, plt_handle=plt, dist_type=dist_type)
plt = plot(n=size[3], dist=single_sample_dist, subplot=324, plt_handle=plt, dist_type=dist_type)
plt = plot(n=size[4], dist=single_sample_dist, subplot=325, plt_handle=plt, dist_type=dist_type)
plt = plot(n=size[5], dist=single_sample_dist, subplot=326, plt_handle=plt, dist_type=dist_type)
plt.tight_layout()
plt.savefig('sum_of_{}_dist_2.png'.format(dist_type), dpi=200)
