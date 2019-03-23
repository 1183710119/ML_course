#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@project: ML_lab2
@author: 王晨懿
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import re
from regression import batch_gradient_descent
from regression import evaluate

if __name__ == '__main__':
    attrs = []
    with open('iris.data', 'r') as f:
        for i in range(100):  # 读取前100行的两类数据
            line = f.readline()
            attrs.append([float(x) for x in line.split(',')[:4]])
    X = np.c_[np.ones(100, ), np.array(attrs)]
    print(X.shape)
    y = [0]*50+[1]*50
    w = batch_gradient_descent(X, y)
    evaluate(X, y, w)
