#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/17 16:15
# @Author  : YangYusheng
# @File    : sklearn_SVM.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import operator
from sklearn import svm

def getData(fileNameStr):
    # encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
    DataDF = pd.read_csv(fileNameStr, encoding="utf-8")
    DataDF = DataDF.fillna(method='ffill')
    DataDF = DataDF.fillna(method='bfill')
    DataDF = DataDF.values
    return DataDF

DataDF = getData("telco.csv")
DataX = DataDF[:,:-1]
DataY = DataDF[:,-1]

clf = svm.SVC()
clf.fit(DataX,DataY)
# print (clf.predict(DataX))

# 验证
loss = DataY - clf.predict(DataX)
count = 0

for i in range(len(DataY)):
    if loss[i]==0:
        count+=1

print(count/len(DataY))
