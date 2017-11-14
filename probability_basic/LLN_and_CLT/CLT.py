# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Created on Sun Nov 12 08:44:37 2017

@author: Belter
"""

# p = 0.999
para = 0.4
size = [1, 4, 20, 80, 200, 10000]


def sampling2pmf(n, p, m=10000):
    """
    n: sample size for each experiment
    m: how many times do you do experiment, fix in 10000
    p: parameter for distribution
    """
    ber_dist = stats.bernoulli(p)  # 定义一个伯努利分布
    sum_of_samples = []
    for i in range(m):
        samples = ber_dist.rvs(size=n)
        sum_of_samples.append(np.sum(samples))
    val, cnt = np.unique(sum_of_samples, return_counts=True)
    pmf = cnt / len(sum_of_samples)
    return val, pmf

# 下面是利用matplotlib画图
plt.figure(1)
# plot bernoulli distribution, n = 1
plt.subplot(321)  # 两行一列，第一个子图
sample_result = sampling2pmf(n=size[0], p=para)
# print(sample_result)
plt.vlines(sample_result[0], 0, sample_result[1],
           colors='g', linestyles='-', lw=3)
plt.ylabel('Probability')
plt.title('PMF of bernoulli dist. (n={})'.format(size[0]))


plt.subplot(322)
sample_result = sampling2pmf(n=size[1], p=para)
# print(sample_result)
plt.vlines(sample_result[0], 0, sample_result[1],
           colors='g', linestyles='-', lw=3)
plt.ylabel('Probability')
plt.title('Sum of bernoulli dist. (n={})'.format(size[1]))


plt.subplot(323)
sample_result = sampling2pmf(n=size[2], p=para)
# print(sample_result)
plt.vlines(sample_result[0], 0, sample_result[1],
           colors='g', linestyles='-', lw=3)
plt.ylabel('Probability')
plt.title('Sum of bernoulli dist. (n={})'.format(size[2]))


plt.subplot(324)
sample_result = sampling2pmf(n=size[3], p=para)
# print(sample_result)
plt.vlines(sample_result[0], 0, sample_result[1],
           colors='g', linestyles='-', lw=3)
plt.ylabel('Probability')
plt.title('Sum of bernoulli dist. (n={})'.format(size[3]))


plt.subplot(325)
sample_result = sampling2pmf(n=size[4], p=para)
# print(sample_result)
plt.vlines(sample_result[0], 0, sample_result[1],
           colors='g', linestyles='-', lw=3)
plt.ylabel('Probability')
plt.title('Sum of bernoulli dist. (n={})'.format(size[4]))


plt.subplot(326)
sample_result = sampling2pmf(n=size[5], p=para)
# print(sample_result)
plt.vlines(sample_result[0], 0, sample_result[1],
           colors='g', linestyles='-', lw=3)
plt.ylabel('Probability')
plt.title('Sum of bernoulli dist. (n={})'.format(size[5]))

plt.tight_layout()
# plt.show()
plt.savefig('sum_of_ber_dist.png', dpi=200)
