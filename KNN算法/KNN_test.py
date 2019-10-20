#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/10 8:49
# @Author  : YangYusheng
# @File    : KNN_test.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import operator

def createDataSet():
    group = np.array([[1,1,1,1,1],
                      [0,0,0,1,1],
                      [1,1,0,1,0],
                      [1,0,1,0,0],
                      [0,1,0,1,0],
                      [0,0,0,1,0],
                      [0,0,1,0,0],
                      [1,0,0,1,1],
                      [0,1,0,1,1],
                      [1,1,0,1,1]])
    labels = ['链球菌喉炎','过敏','感冒','链球菌喉炎','感冒','过敏','链球菌喉炎','过敏','感冒','感冒']
    return group, labels

def getDataY():
    fileNameStr = 'D:/Work/数据挖掘ML/KNN算法/test_Y.csv'

    # encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
    DataDF = pd.read_csv(fileNameStr, encoding="utf-8")
    DataDF = DataDF.values
    DataDF = np.mat(DataDF)
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
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


dataX,dataY = createDataSet()
a = [0,0,1,1,1]
print(type(a))
print(type(dataX))
print(type(dataY))
first = KNN([0,0,1,1,1],dataX, dataY, 3)
# print(first)
# second = KNN([1,1,0,0,1],dataX, dataY, 3)
# print(second)


