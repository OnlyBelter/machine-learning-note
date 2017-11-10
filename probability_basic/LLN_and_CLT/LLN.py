import random
import matplotlib.pyplot as plt


def flip_plot(min_exp, max_exp):
    """
    Assumes min_exp and min_exp positive integers; min_exp < max_exp
    Plots results of 2**min_exp to 2**max_exp coin flips
    抛硬币的次数为2的min_exp次方到2的max_exp次方
    一共进行了 2**max_exp - 2**min_exp 轮实验，每轮实验抛硬币次数逐渐增加
    """

    ratios = []
    x_axis = []
    for exp in range(min_exp, max_exp + 1):
        x_axis.append(2**exp)
    for numFlips in x_axis:
        num_heads = 0  # 初始化，硬币正面朝上的计数为0
        for n in range(numFlips):
            if random.random() < 0.5:  # random.random()从[0, 1)随机的取出一个数
                num_heads += 1  # 当随机取出的数小于0.5时，正面朝上的计数加1
        num_tails = numFlips - num_heads  # 得到本次试验中反面朝上的次数
        ratios.append(num_heads/float(num_tails))  # 正反面计数的比值
    plt.title('Heads/Tails Ratios')
    plt.xlabel('Number of Flips')
    plt.ylabel('Heads/Tails')
    plt.plot(x_axis, ratios)
    plt.hlines(1, 0, x_axis[-1], linestyles='dashed', colors='r')
    plt.show()

flip_plot(4, 16)

