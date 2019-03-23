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


# 随机生成k类样本，每类样本有n个
def generate_data(k, n):
    x1, x2 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], n).T
    data = np.c_[np.array(x1).reshape(n, ), np.array(x2).reshape(n, )]
    for i in range(k - 1):
        mean = np.random.randint(2, high=20, size=2)
        cov = [[random.random(), 0], [0, random.random()]]
        x1, x2 = np.random.multivariate_normal(mean, cov, n).T
        X = np.c_[np.array(x1).reshape(n, ), np.array(x2).reshape(n, )]
        data = np.r_[data, X]
    return data


# 计算欧氏距离
def dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


# 返回最小值及其索引
def my_min(l):
    min_val, min_idx = float('inf'), 0
    for i in range(len(l)):
        if l[i] < min_val:
            min_val = l[i]
            min_idx = i
    return min_val, min_idx


# K-means
def k_means(data, k):
    """
    k-means算法
    :param data: 样本矩阵
    :param k: 类别数
    :return: 分类结果
    """
    N, m = data.shape
    means = [data[random.randint(0, N - 1)] for i in range(k)]
    for i in range(k):
        means[i] = 20 * np.random.random(2)
    y = [-1] * N  # 初始化分类
    update = True
    C = [-1] * k
    while update:
        update = False
        for i in range(len(data)):
            d, idx = my_min([dist(data[i], mean) for mean in means])
            y[i] = idx
        for i in range(k):
            C[i] = [data[j] for j in range(N) if y[j] == i]  # 第j个样本属于第i类

        for i in range(k):
            # 第i类中的样本C[j]
            if len(C[i]) == 0:
                continue
            sum = np.zeros(m)
            for j in range(len(C[i])):
                sum += C[i][j]
            new_mean = sum / len(C[i])  # 新的均值
            if abs(np.max(means[i]) - np.max(new_mean)) > math.exp(-10):
                means[i] = new_mean
                update = True  # 如果所有均值向量均未更新，则update=False，退出循环
    return C, means


def main(k, n):
    """
    :param k: 生成数据的类别
    :param n: 每类数据的个数
    """
    data = generate_data(k, n)
    C, means = k_means(data, k)
    for mean in means:
        plt.plot(mean[0], mean[1], 'x')
    for i in range(k):
        cluster = C[i]
        print('The size of C[%d]:\t%d' % (i, len(np.array(cluster))))
        if not cluster:
            continue
        x = np.array(cluster)[:, 0]
        y = np.array(cluster)[:, 1]
        plt.plot(x, y, '.')
    plt.show()


if __name__ == '__main__':
    main(5, 20)
