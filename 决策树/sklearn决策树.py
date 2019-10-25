#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/24 13:44
# @Author  : YangYusheng
# @File    : sklearn决策树.py
# @Software: PyCharm

import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("german_clean.csv")

data = data.values

X_train = data[:-500,:-1]
X_test = data[-500:,:-1]
y_train = data[:-500,-1]
y_test = data[-500:,-1]

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print("训练集准确率")
print('Train score:{:.3f}'.format(tree.score(X_train,y_train)))
print("测试集准确率")
print('Test score:{:.3f}'.format(tree.score(X_test,y_test)))