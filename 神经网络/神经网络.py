#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/2 19:23
# @Author  : YangYusheng
# @File    : 神经网络.py
# @Software: PyCharm

import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 读取文件
X = pd.read_csv("X_data.csv",header=None)
y = pd.read_csv("y_label.csv",header=None)
theta1 = pd.read_csv("Theta1.csv",header=None)
theta2= pd.read_csv("Theta2.csv",header=None)

# theta1
X = np.hstack([np.ones((len(X), 1)) , X])
k = X.dot(theta1.T)
k = sigmoid(k)

# theta2
k = np.hstack([np.ones((len(k), 1)) , k])
k = k.dot(theta2.T)

index = np.argmax(k,axis=1)
index=index+1

y_2 = []
for i in range(len(y)):
    y_2.append(y[0][i])

y = y_2
y = np.array(y)

loss = y-index

count = 0

for i in range(len(y)):
    if abs(loss[i] - 0) < 0.0001:
        count += 1

print(count / len(y))

