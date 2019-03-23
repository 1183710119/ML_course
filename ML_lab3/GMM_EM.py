#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@project: ML_lab3
@author: 王晨懿
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from K_means import generate_data


# 计算高斯分布概率密度函数
def gauss_prob(x, mu, cov):
    m = len(x)
    power = math.exp(-1 / 2 * np.dot((x - mu).reshape(1, m), np.linalg.inv(cov).dot((x - mu).reshape(m, 1))))
    temp = pow(2 * math.pi, m / 2) * pow(np.linalg.det(cov), 0.5)
    return power / temp


# 返回最小值的索引
def my_max(l):
    max_val, max_idx = float('-inf'), 0
    for i in range(len(l)):
        if l[i] > max_val:
            max_val = l[i]
            max_idx = i
    return max_idx


def gmm_em(data, k):
    """
    EM算法实现GMM模型
    :param data: 样本矩阵
    :param k: 类别数
    :return: 分类结果及各项参数
    """
    n, m = data.shape
    pi = [1 / k] * k
    mu = [data[i * int(n / k)] for i in range(k)]
    # mu = [data[i] for i in range(k)]
    # mu = [data[random.randint(0, n - 1)] for i in range(k)]
    cov = [np.eye(m) for i in range(k)]
    gamma = np.zeros((n, k))
    pre = float('inf')

    # EM 算法
    for time in range(100):
        # E 步骤
        for i in range(n):
            temp = [pi[j] * gauss_prob(data[i], mu[j], cov[j]) for j in range(k)]
            sum_temp = sum(temp)
            for j in range(k):
                gamma[i][j] = temp[j] / sum_temp

        # M 步骤
        temp = [sum([gamma[i][j] for i in range(n)]) for j in range(k)]
        for j in range(k):
            mu[j] = sum([gamma[i][j] * data[i] for i in range(n)]) / temp[j]
            cov[j] = sum([gamma[i][j] * np.dot((data[i] - mu[j]).reshape(m, 1), (data[i] - mu[j]).reshape(1, m))
                          for i in range(n)]) / temp[j]
            pi[j] = temp[j] / n

        # 检查对数似然函数的收敛性
        log_lik_func = 0
        for i in range(n):
            log_lik_func += math.log(sum([pi[j] * gauss_prob(data[i], mu[j], cov[j]) for j in range(k)]))
        print(time, log_lik_func)
        if abs(log_lik_func - pre) < math.exp(-10):
            break
        else:
            pre = log_lik_func

    # 分类
    y = [my_max(gamma[i]) for i in range(n)]
    C = [[data[i] for i in range(n) if y[i] == j] for j in range(k)]

    # 输出
    for i in range(k):
        print()
        print('The size of C[%d]:\t%d' % (i, len(C[i])))
        if not C[i]:
            continue
        print('mu =', np.array(mu[i]))
        print('cov =', np.array(cov[i]))

    return C, pi, mu, cov


def test_manual_data(k, n):
    data = generate_data(k, n)
    C, pi, mu, cov = gmm_em(data, k)
    for i in range(k):
        plt.plot(mu[i][0], mu[i][1], 'x')
        if not C[i]:
            continue
        x = np.array(C[i])[:, 0]
        y = np.array(C[i])[:, 1]
        plt.plot(x, y, '.')
    plt.show()


def test_uci_data(datapath):
    data = []
    with open(datapath, 'r') as f:
        for i in range(150):
            line = f.readline()
            data.append([float(x) for x in line.split(',')[:4]])
    gmm_em(np.array(data), 3)


if __name__ == '__main__':
    test_manual_data(5, 20)
    # test_uci_data('iris.data')
