#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@project: ML_Lab1
@author: 王晨懿
"""

import random
import math
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
# 子图1：最小二乘法求解析解
plot_ls = fig.add_subplot(2, 2, 1)  # 三个参数分别为：行数，列数，本子图是所有子图中的第几个
plot_ls.set_title('Analytical solution')
# 子图2：共轭梯度法
plot_conjgrad = fig.add_subplot(2, 2, 2)
plot_conjgrad.set_title('Conjugate gradient method')
# 子图3：梯度下降法
plot_bgd = fig.add_subplot(2, 2, 3)
plot_bgd.set_title('Gradient descent')
# 子图4：损失函数的值随着迭代次数的变化关系
plot_bgd_loss = fig.add_subplot(2, 2, 4)
plot_bgd_loss.set_title('relationship between loss - iter_count')


# 生成数据
def generate_data(start=0, end=1, N=100, sigma=0.1):  # sigma 高斯噪声方差
    # 在0-2*pi的区间上生成N个点作为输入数据
    x = np.linspace(start, end, N)
    y = np.sin(2 * np.pi * x)
    print(x)

    # 对输入数据加入gauss噪声
    for i in range(N):
        y[i] += random.gauss(0, sigma)  # 高斯噪声
    # 画出点
    plot_ls.plot(x, y, 'o')
    plot_conjgrad.plot(x, y, 'o')
    plot_bgd.plot(x, y, 'o')
    return x, y


# 最小二乘法
def LeastSquares(x_vector, y_vector, m, regula=0, lamda=math.exp(-10)):
    (N,) = x_vector.shape
    x_list = [x_vector[i] ** n for i in range(N) for n in range(m + 1)]
    X = np.array(x_list).reshape(N, m + 1)
    X_transpose = X.transpose()
    T = y_vector.reshape(N, 1)

    # 计算W
    a = np.dot(X_transpose, X)
    if regula == 1:
        a += np.eye(m + 1) * lamda
    w = np.dot(np.linalg.inv(a), np.dot(X_transpose, T))
    y_predict = np.dot(X, w).reshape(N, )
    if regula == 0:  # 无正则项
        plot_ls.plot(x_vector, y_predict, 'b', label='without regularization term')
    else:  # 有正则项
        plot_ls.plot(x_vector, y_predict, 'r', label='with regularization term')

    loss = sum([pow(np.dot(X[i], w) - y_vector[i], 2) for i in range(N)]) / 2 / m
    if regula == 1:
        loss += 1 / 2 * lamda * np.dot(w.transpose(), w).reshape(1)
    return w.flatten(), loss


# 梯度下降法
def bgd(x_vector, y_vector, m, step_size=0.7, max_iter_count=50000, regula=0, lamda=math.exp(-11)):
    (N,) = x_vector.shape
    w = np.zeros((m + 1,))
    x_list = [x_vector[i] ** n for i in range(N) for n in range(m + 1)]
    X = np.array(x_list).reshape(N, m + 1)

    loss = 10
    iter_count = 0
    temp = [0] * (m + 1)
    c = [0] * (int)(max_iter_count / 100 + 1)
    l = [0] * (int)(max_iter_count / 100 + 1)
    if regula == 0:  # 如果不加正则项
        lamda = 0
    while loss > 0.001 and iter_count < max_iter_count:
        # print(W)
        error = [np.dot(X[j], w) - y_vector[j] for j in range(N)]
        # print(error)
        for i in range(m + 1):
            temp[i] = (1 - lamda) * w[i] - step_size / N * sum([error[j] * X[j][i] for j in range(N)])
        for i in range(m + 1):
            w[i] = temp[i]

        loss = 1 / (2 * N) * sum([pow(error[j], 2) for j in range(N)])
        print("iter_count: ", iter_count, "the loss:", loss)
        if iter_count % 100 == 0:
            print("iter_count: ", iter_count, "the loss:", loss)
            t = (int)(iter_count / 100)
            c[t] = iter_count
            l[t] = loss
        iter_count += 1

    y_predict = np.dot(X, w).reshape(N, )
    t = (int)(iter_count / 100)
    if regula == 0:
        plot_bgd.plot(x_vector, y_predict, 'b', label='without regularization term')
        plot_bgd_loss.plot(c[:t], l[:t], 'b')
    elif regula == 1:
        plot_bgd.plot(x_vector, y_predict, 'r', label='with regularization term')
        plot_bgd_loss.plot(c[:t], l[:t], 'r')
        loss += 1 / 2 * lamda * np.dot(w.transpose(), w)
    return w.flatten(), loss


# 共轭梯度法
def conjgrad(x_vector, y_vector, m, regula=0, lamda=math.exp(-7)):
    (N,) = x_vector.shape
    x_list = [x_vector[i] ** n for i in range(N) for n in range(m + 1)]
    X = np.array(x_list).reshape(N, m + 1)
    X_transpose = X.transpose()
    A = np.dot(X_transpose, X)
    if regula == 1:  # 如果有正则项
        A += np.eye(m + 1) * lamda
    b = np.dot(X_transpose, y_vector.reshape(N, 1))  # (m+1)*1

    w = np.zeros((m + 1, 1))
    r = b - np.dot(A, w)  # 初始误差
    p = r  # 共轭向量
    rsold = np.dot(r.transpose(), r)
    iter_count = 0
    while 1:
        Ap = np.dot(A, p)  # (m+1)*1
        alpha = rsold / np.dot(p.transpose(), Ap)
        w = w + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.transpose(), r)
        if pow(rsnew, 2) < 1e-10:
            break
        p = r + rsnew / rsold * p
        rsold = rsnew
        iter_count += 1

        loss = rsold / 2 / m
        if regula == 1:
            loss += 1 / 2 * lamda * np.dot(w.transpose(), w)
        print("iter_count: ", iter_count, "the loss:", loss)
    y_predict = np.dot(X, w).reshape(N, )
    if regula == 0:
        plot_conjgrad.plot(x_vector, y_predict, 'b', label='without regularization term')
    elif regula == 1:
        plot_conjgrad.plot(x_vector, y_predict, 'r', label='with regularization term')

    return w.flatten(), loss


if __name__ == '__main__':
    m = 9
    x, y = generate_data(N=10)

    # 用解析解求解两种loss的最优解
    (w1, l1) = LeastSquares(x, y, m)  # 无正则项
    (w2, l2) = LeastSquares(x, y, m, regula=1)  # 有正则项
    print('解析解 无正则项： W =', w1, '\n\tloss =', l1)
    print('解析解 有正则项： W =', w2, '\n\tloss =', l2)

    # 共轭梯度
    (w3, l3) = conjgrad(x, y, m)  # 无正则项
    (w4, l4) = conjgrad(x, y, m, regula=1)  # 有正则项
    print('共轭梯度法 无正则项： W =', w3, '\n\tloss =', l3)
    print('共轭梯度法 有正则项： W =', w4, '\n\tloss =', l4)

    # # 梯度下降
    # (w5, l5) = bgd(x, y, m)  # 无正则项
    # (w6, l6) = bgd(x, y, m, regula=1)  # 有正则项
    # print('梯度下降法 无正则项： W =', w5, '\n\tloss =', l5)
    # print('梯度下降法 有正则项： W =', w6, '\n\tloss =', l6)

    plt.show()
