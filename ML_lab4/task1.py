#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@project: ML_lab4
@author: 王晨懿
@student ID: 1162100102
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def generate_data(n, angle):
    """
    生成数据
    :param n: 样本数
    :param angle: 绕x轴旋转的角度
    :return: 数据集
    """
    x, y, z = np.random.multivariate_normal([0, 0, 0], [[20, 0, 0], [0, 20, 0], [0, 0, 1]], n).T
    X = np.c_[np.array(x).reshape(n, ), np.array(y).reshape(n, ), np.array(z).reshape(n, )]
    X = rotate_mtx(X, angle)
    return X


def rotate_mtx(X, angle):
    """
    将样本绕x轴旋转
    :param X: 样本集
    :param angle: 旋转角度
    :return: 旋转后的样本集
    """
    mtx = np.array([[1, 0, 0], [0, math.cos(angle), -1 * math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])
    return np.dot(X, mtx)


def pca(data, k):
    n, m = data.shape
    # 中心化
    mu = np.mean(data, axis=0)
    X = np.array([x - mu for x in data])
    # 协方差矩阵
    # sigma = np.dot(X.T, X) / n-1  # 无偏估计，除以n-1
    sigma = np.cov(X, rowvar=0)  # 求协方差矩阵,rowvar为0，一行代表一个样本

    # 获取前k个特征向量
    eig_val, eig_vec = np.linalg.eig(sigma)  # 特征根和特征向量
    eig_pairs = [(abs(eig_val[i]), eig_vec[:, i]) for i in range(m)]
    eig_pairs.sort(reverse=True, key=lambda item: item[0])  # 根据特征值降序排列
    reduce = np.array([eig_pairs[i][1] for i in range(k)]).T

    # 将数据转移到新坐标系
    reduce_d_data = np.dot(X, reduce)
    # 用主成分对样本进行近似
    new_data = np.dot(reduce_d_data, reduce.T) + mu
    return reduce, new_data


def test_manually_data(n=50):
    data = generate_data(n, math.pi / 6)
    ax = plt.subplot(111, projection='3d')
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-12, 12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    ax.scatter(x, y, z, c='b')

    reduce, new_data = pca(data, 2)
    print(reduce)

    x, y, z = new_data[:, 0], new_data[:, 1], new_data[:, 2]
    ax.scatter(x, y, z, c='r', marker='x')
    plt.show()


if __name__ == '__main__':
    test_manually_data()
