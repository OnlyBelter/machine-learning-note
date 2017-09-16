import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def uniform_distribution():
    # 均匀分布
    # 两个参数：loc表示该分布的起点，scale表示区间长度
    uniform_dis = stats.uniform(loc=2, scale=4)
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
    print(np.allclose([0.001, 0.5, 0.999], uniform_dis.cdf(vals)))

    r = uniform_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    plt.show()
uniform_distribution()

