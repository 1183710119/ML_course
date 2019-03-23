#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@project: ML_lab4
@author: 王晨懿
@student ID: 1162100102
"""

import os
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from task1 import pca


# 读入minst数据集
def read_mnist_dataset(path='MNIST_data'):
    labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    images_path = os.path.join(path, 'train-images.idx3-ubyte')
    with open(labels_path, 'rb') as f:
        struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    with open(images_path, 'rb') as f:
        struct.unpack('>IIII', f.read(16))
        imgs = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)
    return imgs


# 展示图片
def show_image(images):
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):  # 只取前十个展示
        img = images[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])


# 计算信噪比
def evaluate(before, after):
    signal = np.sum(np.power(after, 2))
    noise = np.sum(np.power(before - after, 2))
    snr = 10 * math.log(signal / noise, 10)
    return snr


# 测试mnist数据
def test_mnist(dimension=100):
    imgs = read_mnist_dataset()
    origin_imgs = imgs[:1000]  # 取前1000张图片进行测试

    reduce, new_data = pca(origin_imgs, dimension)

    # 展示降维前后图片，只展示前10张
    show_image(origin_imgs)
    show_image(np.array(new_data, dtype=float))

    # 计算信噪比
    snr = evaluate(origin_imgs, new_data)
    print('SNR =', snr)


if __name__ == '__main__':
    test_mnist()
    plt.tight_layout()
    plt.show()
