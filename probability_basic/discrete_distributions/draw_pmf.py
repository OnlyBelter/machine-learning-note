from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


def bernoulli_pmf(p=0.0):
    """
    伯努利分布，只有一个参数
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html#scipy.stats.bernoulli
    :param p: 试验成功的概率，或结果为1的概率
    :return:
    """
    ber_dist = stats.bernoulli(p)
    x = [0, 1]
    x_name = ['0', '1']
    pmf = [ber_dist.pmf(x[0]), ber_dist.pmf(x[1])]
    plt.bar(x, pmf, width=0.15)
    plt.xticks(x, x_name)
    plt.ylabel('Probability')
    plt.title('PMF of bernoulli distribution')
    plt.show()

# bernoulli_pmf(p=0.3)


def binom_pmf(n=1, p=0.1):
    """
    二项分布有两个参数
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html#scipy.stats.binom
    :param n:试验次数
    :param p:单次实验成功的概率
    :return:
    """
    binom_dis = stats.binom(n, p)
    x = np.arange(binom_dis.ppf(0.0001), binom_dis.ppf(0.9999))
    print(x)  # [ 0.  1.  2.  3.  4.]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, binom_dis.pmf(x), 'bo', label='binom pmf')
    ax.vlines(x, 0, binom_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of binomial distribution(n={}, p={})'.format(n, p))
    plt.show()

# binom_pmf(n=20, p=0.6)


def poisson_pmf(mu=3):
    """
    泊松分布
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson
    :param mu: 单位时间（或单位面积）内随机事件的平均发生率
    :return:
    """
    poisson_dis = stats.poisson(mu)
    x = np.arange(poisson_dis.ppf(0.001), poisson_dis.ppf(0.999))
    print(x)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, poisson_dis.pmf(x), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of poisson distribution(mu={})'.format(mu))
    plt.show()

# poisson_pmf(mu=8)


def plot_bar():
    y = [1, 2, 3, 4, 5]
    x_name = ['apple', 'orange', 'pear', 'mango', 'peach']
    x = np.arange(len(x_name))
    plt.bar(x, y)
    plt.xticks(x, x_name)
    plt.show()

# plot_bar()


def custom_made_discrete_dis_pmf():
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    :return:
    """
    xk = np.arange(7)  # 所有可能的取值
    print(xk)  # [0 1 2 3 4 5 6]
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)  # 各个取值的概率
    custm = stats.rv_discrete(name='custm', values=(xk, pk))

    X = custm.rvs(size=20)
    print(X)

    fig, ax = plt.subplots(1, 1)
    ax.plot(xk, custm.pmf(xk), 'ro', ms=8, mec='r')
    ax.vlines(xk, 0, custm.pmf(xk), colors='r', linestyles='-', lw=2)
    plt.title('Custom made discrete distribution(PMF)')
    plt.ylabel('Probability')
    plt.show()

# custom_made_discrete_dis_pmf()


def sampling_and_empirical_dis():
    xk = np.arange(7)  # 所有可能的取值
    print(xk)  # [0 1 2 3 4 5 6]
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)  # 各个取值的概率
    custm = stats.rv_discrete(name='custm', values=(xk, pk))

    X1 = custm.rvs(size=20)  # 第一次抽样
    X2 = custm.rvs(size=200)  # 第二次抽样
    # 计算X1＆X2中各个结果出现的频率(相当于PMF)
    val1, cnt1 = np.unique(X1, return_counts=True)
    val2, cnt2 = np.unique(X2, return_counts=True)
    pmf_X1 = cnt1 / len(X1)
    pmf_X2 = cnt2 / len(X2)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(xk, custm.pmf(xk), 'ro', ms=8, mec='r', label='theor. pmf')
    plt.vlines(xk, 0, custm.pmf(xk), colors='r', lw=5, alpha=0.2)
    plt.vlines(val1, 0, pmf_X1, colors='b', linestyles='-', lw=3, label='X1 empir. pmf')
    plt.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('Theoretical dist. PMF vs Empirical dist. PMF')
    plt.subplot(212)
    plt.plot(xk, custm.pmf(xk), 'ro', ms=8, mec='r', label='theor. pmf')
    plt.vlines(xk, 0, custm.pmf(xk), colors='r', lw=5, alpha=0.2)
    plt.vlines(val2, 0, pmf_X2, colors='g', linestyles='-', lw=3, label='X2 empir. pmf')
    plt.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.show()

sampling_and_empirical_dis()
