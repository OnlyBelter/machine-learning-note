import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def uniform_distribution(loc=0, scale=1):
    """
    均匀分布，在实际的定义中有两个参数，分布定义域区间的起点和终点[a, b]
    :param loc: 该分布的起点, 相当于a
    :param scale: 区间长度, 相当于 b-a
    :return:
    """
    uniform_dis = stats.uniform(loc=loc, scale=scale)
    x = np.linspace(uniform_dis.ppf(0.01),
                    uniform_dis.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.uniform.pdf(x, loc=2, scale=4), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, uniform_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = uniform_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # Check accuracy of cdf and ppf
    print(np.allclose([0.001, 0.5, 0.999], uniform_dis.cdf(vals)))  # Ture

    r = uniform_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Unif({}, {})'.format(loc, loc+scale))
    ax.legend(loc='best', frameon=False)
    plt.show()

# uniform_distribution(loc=2, scale=4)


def exponential_dis(loc=0, scale=1.0):
    """
    指数分布，exponential continuous random variable
    按照定义，指数分布只有一个参数lambda，这里的scale = 1/lambda
    :param loc: 定义域的左端点，相当于将整体分布沿x轴平移loc
    :param scale: lambda的倒数，loc + scale表示该分布的均值，scale^2表示该分布的方差
    :return:
    """
    exp_dis = stats.expon(loc=loc, scale=scale)
    x = np.linspace(exp_dis.ppf(0.000001),
                    exp_dis.ppf(0.999999), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.expon.pdf(x, loc=loc, scale=scale), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, exp_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = exp_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # Check accuracy of cdf and ppf
    print(np.allclose([0.001, 0.5, 0.999], exp_dis.cdf(vals)))

    r = exp_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Exp(0.5)')
    ax.legend(loc='best', frameon=False)
    plt.show()

# exponential_dis(loc=0, scale=2)


def diff_exp_dis():
    """
    不同参数下的指数分布
    :return:
    """
    exp_dis_0_5 = stats.expon(scale=0.5)
    exp_dis_1 = stats.expon(scale=1)
    exp_dis_2 = stats.expon(scale=2)

    x1 = np.linspace(exp_dis_0_5.ppf(0.001), exp_dis_0_5.ppf(0.9999), 100)
    x2 = np.linspace(exp_dis_1.ppf(0.001), exp_dis_1.ppf(0.999), 100)
    x3 = np.linspace(exp_dis_2.ppf(0.001), exp_dis_2.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, exp_dis_0_5.pdf(x1), 'b-', lw=2, label=r'lambda = 2')
    ax.plot(x2, exp_dis_1.pdf(x2), 'g-', lw=2, label='lambda = 1')
    ax.plot(x3, exp_dis_2.pdf(x3), 'r-', lw=2, label='lambda = 0.5')
    plt.ylabel('Probability')
    plt.title(r'PDF of Exponential Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()

# diff_exp_dis()


def normal_dis(miu=0, sigma=1):
    """
    正态分布有两个参数
    :param miu: 均值
    :param sigma: 标准差
    :return:
    """

    norm_dis = stats.norm(miu, sigma)  # 利用相应的分布函数及参数，创建一个冻结的正态分布(frozen distribution)
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
    plt.text(-5.0, .12, r'$\mu={},\ \sigma={}$'.format(miu, sigma))  # 3是标准差，不是方差
    plt.legend(loc='best', frameon=False)
    # plot cdf
    plt.subplot(212)
    plt.plot(x, cdf, 'r-', label='cdf')
    plt.ylabel('Probability')
    plt.legend(loc='best', frameon=False)

    plt.show()

# normal_dis(miu=5, sigma=3)


def diff_normal_dis():
    """
    不同参数下的指数分布
    :return:
    """
    norm_dis_0 = stats.norm(0, 1)  # 标准正态分布
    norm_dis_1 = stats.norm(0, 0.5)  # 标准正态分布
    norm_dis_2 = stats.norm(0, 2)  # 标准正态分布
    norm_dis_3 = stats.norm(2, 2)  # 标准正态分布

    x0 = np.linspace(norm_dis_0.ppf(1e-8), norm_dis_0.ppf(0.99999999), 1000)
    x1 = np.linspace(norm_dis_1.ppf(1e-10), norm_dis_1.ppf(0.9999999999), 1000)
    x2 = np.linspace(norm_dis_2.ppf(1e-6), norm_dis_2.ppf(0.999999), 1000)
    x3 = np.linspace(norm_dis_3.ppf(1e-6), norm_dis_3.ppf(0.999999), 1000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x0, norm_dis_0.pdf(x0), 'r-', lw=2, label=r'miu=0, sigma=1')
    ax.plot(x1, norm_dis_1.pdf(x1), 'b-', lw=2, label=r'miu=0, sigma=0.5')
    ax.plot(x2, norm_dis_2.pdf(x2), 'g-', lw=2, label=r'miu=0, sigma=2')
    ax.plot(x3, norm_dis_3.pdf(x3), 'y-', lw=2, label=r'miu=2, sigma=2')
    plt.ylabel('Probability')
    plt.title(r'PDF of Normal Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()

diff_normal_dis()
