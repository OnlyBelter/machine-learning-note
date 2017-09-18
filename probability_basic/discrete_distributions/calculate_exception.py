"""
Q1: 袋中有5张标号1,2,3,4,5的卡片，现从中有放回地抽出3张卡片，号码之和X的数学期望为
"""
import numpy as np


def calculate_Q1():
    all_result = [(a, b, c) for a in [1, 2, 3, 4, 5]
     for b in [1, 2, 3, 4, 5]
     for c in [1, 2, 3, 4, 5]]

    all_sum = [sum(x) for x in all_result]

    val, count = np.unique(all_sum, return_counts=True)
    print(val, count)

    pmf = count/len(all_sum)
    print(pmf)
    print(np.multiply(np.array(val), np.array(pmf)))
    E = np.sum(np.multiply(np.array(val), np.array(pmf)))
    print(E)
