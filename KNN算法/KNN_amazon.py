#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/17 8:32
# @Author  : YangYusheng
# @File    : KNN_amazon.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import operator

def getData(fileNameStr):
    # encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
    DataDF = pd.read_csv(fileNameStr, encoding="utf-8",header=None)
    DataDF = DataDF.values
    # DataDF = np.mat(DataDF)
    return DataDF

def KNN(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel[0]] = classCount.get(voteIlabel[0], 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

amazon_X = getData("amazon_X.csv")
amazon_Y = getData("amazon_Y.csv")
amazon_Y = amazon_Y.tolist()
# print(cancer_X.shape)
# print(cancer_X[0])
# print(type(cancer_X))
# print(type(cancer_Y))
# print(type(cancer_X[0,:].tolist()))
pre = []
for i in range(len(amazon_Y)):
    a = KNN(amazon_X[i, :].tolist(), amazon_X, amazon_Y, 5)
    pre.append(a)
#
# print(pre)
# print(len(cancer_Y))
true = []
for i in range(len(amazon_Y)):
    true.append(amazon_Y[i][0])
# print(b)
loss = []
for i in range(len(amazon_Y)):
    loss.append(true[i]-pre[i])

# print(loss)a

count = 0
for i in range(len(amazon_Y)):
    if loss[i]==0:
        count+=1

print(count / len(amazon_Y))