#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/9 19:54
# @Author  : YangYusheng
# @File    : 牛顿法.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import math
# 导入csv数据
# dtype = str,最好读取的时候都以字符串的形式读入，不然可能会使数据失真
# 比如一个0010008的编号可能会读取成10008

fileNameStr = 'D:/Work/数据挖掘ML/Logistic回归/telco.csv'

# encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
DataDF = pd.read_csv(fileNameStr, encoding="utf-8")
DataDF = DataDF.fillna(method='ffill')
DataDF = DataDF.fillna(method='bfill')
DataDF = DataDF.values
# print(DataDF.shape)
DataX = DataDF[:, :-1]
DataX = np.hstack([np.ones((len(DataX), 1)), DataX])
DataY = DataDF[:, -1]


# print(DataX.shape)
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# 牛顿法
def newton_method(X, y, reg_lambda=math.exp(-1), max_iter_count=100):
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


# 输出结果
w = newton_method(DataX, DataY)
print("输出结果为:")
print(w)

# 验证命中率
w = np.mat(w).T
DataX = np.mat(DataX)
DataY = np.mat(DataY).T
# loss = abs(DataY-sigmoid(DataX.dot(w)))

predict = sigmoid(DataX.dot(w))
predict_y = np.array(predict>0.5,dtype='int')
print("命中率为:")
print(np.sum(predict_y == DataY) / len(DataY))
