#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/27 12:59
# @Author  : YangYusheng
# @File    : 决策树信息增益率.py
# @Software: PyCharm

import pandas as pd
from collections import Counter
from math import log
import numpy as np

data = pd.read_csv("german_clean.csv")

data = data.values

X_train = data[:-300,:-1]
X_test = data[-300:,:-1]
y_train = data[:-300,-1]
y_test = data[-300:,-1]

def split(X, y, d, value ):
    index_a = (X[:,d]<=value)
    index_b = (X[:,d]>value)
    return X[index_a], X[index_b], y[index_a],y[index_b]


def entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p)
    return res


def try_split(X, y):
    best_entropy = float('inf')
    best_d, best_v = -1, -1
    for d in range(X.shape[1]):
        sorted_index = np.argsort(X[:, d])
        for i in range(1, len(X)):
            if X[sorted_index[i - 1], d] != X[sorted_index[i], d]:
                v = (X[sorted_index[i - 1], d] + X[sorted_index[i], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                e = entropy(y_l) + entropy(y_r)
                if e < best_entropy:
                    best_entropy, best_d, best_v = e, d, v

    return best_entropy, best_d, best_v

best_entropy, best_d, best_v = try_split(X_train, y_train)
print("best_entropy is ",best_entropy)
print("best_d is ",best_d)