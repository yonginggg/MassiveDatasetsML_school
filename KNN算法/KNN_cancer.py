#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/16 22:11
# @Author  : YangYusheng
# @File    : KNN_cancer.py
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

cancer_X = getData("cancer_X.csv")
cancer_Y = getData("cancer_Y.csv")
cancer_Y = cancer_Y.tolist()
# print(cancer_X.shape)
# print(cancer_X[0])
# print(type(cancer_X))
# print(type(cancer_Y))
# print(type(cancer_X[0,:].tolist()))
pre = []
for i in range(len(cancer_Y)):
    a = KNN(cancer_X[i,:].tolist(),cancer_X, cancer_Y, 4)
    pre.append(a)
#
# print(pre)
# print(len(cancer_Y))
true = []
for i in range(len(cancer_Y)):
    true.append(cancer_Y[i][0])
# print(b)
loss = []
for i in range(len(cancer_Y)):
    loss.append(true[i]-pre[i])

# print(loss)

count = 0
for i in range(len(cancer_Y)):
    if loss[i]==0:
        count+=1

print(count/len(cancer_Y))

