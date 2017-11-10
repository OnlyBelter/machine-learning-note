import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compare_poission_exp():
    """
    This post explained the relation between these two distribution
      - https://stats.stackexchange.com/a/2094/134555
      - P(Xt <= x) = 1 - e^(-lambda * x)
    Now, I suppose lambda=1, just like this example(from wiki, Poisson_distribution):
      - On a particular river, overflow floods occur once every 100 years on average.
    :return:
    """
    x = np.arange(20)
    y1 = 1 - np.power(np.e, -x)  # lambda = 1
    y2 = 1 - np.power(np.e, -0.2*x)  # lambda = 0.2
    y3 = 1 - np.power(np.e, -5*x)  # lambda = 1.5
    print(y1)
    print(y2)
    print(y3)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y1, 'r-', label='lambda=1')
    ax.plot(x, y2, 'g-', label='lambda=0.2')
    ax.plot(x, y3, 'b-', label='lambda=5')
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('CDF of exponential distribution')
    plt.show()

compare_poission_exp()
