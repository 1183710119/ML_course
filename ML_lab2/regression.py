#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@project: ML_lab2
@author: 王晨懿

"""
from collections import Iterable
import matplotlib.pyplot as plt
import numpy as np
import math

fig = plt.figure()
plot_bgd = fig.add_subplot(1, 2, 1)
plot_bgd.set_title('Batch gradient descent')
plot_nm = fig.add_subplot(1, 2, 2)
plot_nm.set_title('Newton method')

bgd_lambda = math.exp(-8)
nm_lambda = math.exp(-1)


# 手工生成二维数据
def generate_data(n, independence=True):
    mean0, mean1 = [3, 3], [7, 9]
    # diagonal covariance
    cov = [[2, 0], [0, 3]] if independence else [[2, 2], [2, 3]]
    x0, y0 = np.random.multivariate_normal(mean0, cov, n).T
    x1, y1 = np.random.multivariate_normal(mean1, cov, n).T
    plot_bgd.plot(x0, y0, 'x')
    plot_bgd.plot(x1, y1, '.')
    plot_nm.plot(x0, y0, 'x')
    plot_nm.plot(x1, y1, '.')
    return x0, y0, x1, y1


# 批梯度下降
def batch_gradient_descent(X, y, reg_lambda=bgd_lambda, step_size=0.5, max_iter_count=10000):
    (n, m) = X.shape
    w = np.zeros((m,))
    for i in range(max_iter_count):
        z = sigmoid(np.dot(X, w))
        w = (1 - reg_lambda) * w - step_size / n * np.dot(X.transpose(), z - y)
    return w


# 牛顿法
def newton_method(X, y, reg_lambda=nm_lambda, max_iter_count=1000):
    (n, m) = X.shape
    w = np.zeros((m,))
    for i in range(max_iter_count):
        temp = sigmoid(X.dot(w))
        gradient = X.T.dot(temp - y)
        A = np.eye(n)
        for j in range(n):
            h = sigmoid(X[j].dot(w))
            A[j, j] = h * (1 - h) + 0.0001
        Hessian = X.T.dot(A).dot(X)
        delta_theta = np.linalg.solve(Hessian, gradient) + reg_lambda * w
        # newton's method parameter update
        w = w - delta_theta
    return w


def sigmoid(x):
    if isinstance(x, Iterable):
        return np.array(list([1 / (1 + math.exp(-1 * z)) for z in x]))
    return 1 / (1 + math.exp(-1 * x))


def evaluate(X, y, w):
    (n, m) = X.shape
    y_predict = [0 if x >= 0 else 1 for x in np.dot(X, w)]
    correct_num = sum(map(lambda x: 1 if x[0] == x[1] else 0, zip(y, y_predict)))
    loss = -1 / n * (sum([y[i] * math.log(sigmoid(np.dot(X[i], w)), math.e) +
                          (1 - y[i]) * math.log(sigmoid(1 - np.dot(X[i], w)), math.e) for i in range(n)]))
    print('\t\tCorrect rate =', correct_num / n)
    print('\t\tw =', w)
    print('\t\tloss =', loss)


# 手工生成数据测试
def test_manually_data(func, x1_class0, x2_class0, x1_class1, x2_class1):
    (n,) = x1_class0.shape
    x_class0 = np.c_[np.ones(n, ), np.array(x1_class0).reshape(n, ), np.array(x2_class0).reshape(n, )]
    x_class1 = np.c_[np.ones(n, ), np.array(x1_class1).reshape(n, ), np.array(x2_class1).reshape(n, )]
    X = np.r_[x_class0, x_class1]
    print(X)
    y = np.array([0] * n + [1] * n).reshape(2 * n, )

    x = np.arange(min(x1_class0), max(x1_class1), 0.1)
    plot = plot_bgd if func.__name__ == 'batch_gradient_descent' else plot_nm
    print(func.__name__, '\n\twithout regularization')
    w = func(X, y, reg_lambda=0)
    evaluate(X, y, w)
    plot.plot(x, -1 * (w[0] + w[1] * x) / w[2], 'b')
    print('\twith regularization')
    w = func(X, y)
    evaluate(X, y, w)
    plot.plot(x, -1 * (w[0] + w[1] * x) / w[2], 'r')
    print()


# UCI数据测试
def test_uci_data(func, datapath):
    data = []
    with open(datapath, 'r') as f:
        for i in range(100):  # 读取前100行的两类数据
            line = f.readline()
            data.append([float(x) for x in line.split(',')[:4]])
    X = np.c_[np.ones(100, ), np.array(data)]
    y = [0] * 50 + [1] * 50
    print(func.__name__, '\n\twithout regularization')
    w = func(X, y, reg_lambda=0)
    evaluate(X, y, w)
    print('\twith regularization')
    w = func(X, y)
    evaluate(X, y, w)
    print()


if __name__ == '__main__':
    # x1_class0, x2_class0, x1_class1, x2_class1 = generate_data(20, independence=False)  # 不满足条件独立性假设
    x1_class0, x2_class0, x1_class1, x2_class1 = generate_data(20)  # 满足朴素贝叶斯假设
    print('=========================== test with manually data ===========================')
    test_manually_data(batch_gradient_descent, x1_class0, x2_class0, x1_class1, x2_class1)
    test_manually_data(newton_method, x1_class0, x2_class0, x1_class1, x2_class1)
    # print('\n============================= test with UCI data =============================')
    # test_uci_data(batch_gradient_descent, 'iris.data')
    # test_uci_data(newton_method, 'iris.data')

    plt.show()
