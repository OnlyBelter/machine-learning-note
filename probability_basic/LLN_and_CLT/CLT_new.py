# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Created on Sun Nov 17 18:44:37 2017

@author: Belter
"""


def sampling2pmf(n, dist, t=100000):
    """
    n: sample size for each experiment
    t: how many times do you do experiment, fix in 100000
    dist: frozen distribution
    """
    current_dist = dist
    sum_of_samples = np.zeros(t)
    for i in range(t):
        samples = []
        for j in range(n):  # n次独立的试验
            samples.append(current_dist.rvs())
        sum_of_samples[i] = np.sum(samples)
    return sum_of_samples


def plot(n, dist, subplot):
    """
    :param n: sample size
    :param dist: distribution of each single sample
    :param subplot: location of sub-graph, such as 221, 222, 223, 224
    """
    plt.subplot(3, 2, subplot)
    mu = n * dist.mean()
    sigma = np.sqrt(n * dist.var())
    samples = sampling2pmf(n=n, dist=dist)
    # normed参数可以对直方图进行标准化，从而使纵坐标表示概率而不是次数
    plt.hist(samples, normed=True, bins=100, color='#348ABD',
             label='{} RVs'.format(n))
    plt.ylabel('Probability')
    # normal distribution
    norm_dis = stats.norm(mu, sigma)
    norm_x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 10000)
    pdf = norm_dis.pdf(norm_x)
    plt.plot(norm_x, pdf, 'r--', alpha=0.6, label='N(${0:.0f}, {1:.2f}^2$)'.format(mu, sigma))
    plt.legend(loc='upper left', prop={'size': 8})

size = [1, 2, 3, 4, 8, 10]

# sum of uniform distribution
dist_type = 'uniform'
uniform_para = [3, 2]
single_sample_dist = stats.uniform(loc=uniform_para[0], scale=uniform_para[1])  # 定义一个均匀分布

# 下面是利用matplotlib画图
plt.figure(figsize=(10, 7))  # bigger size
plt.suptitle('Sum of {} dist. random variables (RVs) converge to a Gaussian distribution (CLT)'.format(dist_type),
             fontsize=16)
for s in range(len(size)):
    plot(n=size[s], dist=single_sample_dist, subplot=s+1)

plt.savefig('sum_of_{}_dist.png'.format(dist_type), dpi=200)
