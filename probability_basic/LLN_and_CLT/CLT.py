import random
import numpy as np
import matplotlib.pyplot as plt


def flip_plot(n, p):
    """
    Assumes min_exp and min_exp positive integers; min_exp < max_exp
    Plots results of 2**min_exp to 2**max_exp coin flips
    抛硬币的次数为2的min_exp次方到2的max_exp次方
    一共进行了 2**max_exp - 2**min_exp 轮实验，每轮实验抛硬币次数逐渐增加
    """

    delta_p = []  # delta p for each person's result
    x_axis = []  # 每个人抛硬币的次数，其长度表示试验人数
    # X_mean = n * p  # 理论均值
    for i in range(int(np.sqrt(n))):
        x_axis.append(n)
    for numFlips in x_axis:
        num_heads = 0  # 初始化，硬币正面朝上的计数为0
        for n in range(numFlips):
            if random.random() < p:  # random.random()从[0, 1)随机的取出一个数
                num_heads += 1  # 当随机取出的数小于p时，正面朝上的计数加1
        num_tails = numFlips - num_heads  # 得到本次试验中反面朝上的次数
        delta_p.append(num_heads/float(n) - p)  # 每个人的估计值与真实值之间的误差
    # plt.title('Heads/Tails Ratios')
    # plt.xlabel('Number of Flips')
    # plt.ylabel('Heads/Tails')
    # plt.plot(x_axis, ratios)
    # plt.hlines(1, 0, x_axis[-1], linestyles='dashed', colors='r')
    # plt.show()
    return sum(delta_p)  # 误差和

result = []
for j in range(10000):
    result.append(flip_plot(10000, 0.5))
plt.hist(result, bins=100)
plt.show()


