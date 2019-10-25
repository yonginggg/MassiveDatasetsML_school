#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/22 10:54
# @Author  : YangYusheng
# @File    : logistic_cancer.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def getData(fileNameStr):
    # encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
    DataDF = pd.read_csv(fileNameStr, encoding="utf-8",header=None)
    DataDF = DataDF.values
    # DataDF = np.mat(DataDF)
    return DataDF
cancer_X = getData("cancer_X.csv")
cancer_Y = getData("cancer_Y.csv")

classifier = LogisticRegression()
classifier.fit(cancer_X, cancer_Y)

w = classifier.predict(cancer_X)
print("运行结果是:")
print(w)

loss = w-cancer_Y.T

count = 0
for i in range(len(loss)):
    if int(loss[:,i])-0<0.1:
        count+=1

print("准确率为:")
print(count/len(loss))