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

bernoulli_pmf(p=0.3)




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
    xk = np.arange(7)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
    custm = stats.rv_discrete(name='custm', values=(xk, pk))

    fig, ax = plt.subplots(1, 1)
    ax.plot(xk, custm.pmf(xk), 'ro', ms=8, mec='r')
    ax.vlines(xk, 0, custm.pmf(xk), colors='r', linestyles='-', lw=2)
    plt.title('Custom made discrete distribution(PMF)')
    plt.ylabel('Probability')
    plt.show()
