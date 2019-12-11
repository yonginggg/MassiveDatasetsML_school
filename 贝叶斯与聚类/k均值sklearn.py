#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/19 20:29
# @Author  : YangYusheng
# @File    : k均值sklearn.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/19 19:21
# @Author  : YangYusheng
# @File    : k均值.py
# @Software: PyCharm

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


origin = pd.read_csv("DRUG1n.csv")
origin = pd.get_dummies(origin)
print(origin.columns)

clf = KMeans(n_clusters=5)  # 聚类算法，参数n_clusters=3，聚成3类
y_pred = clf.fit_predict(origin)  # 直接对数据进行聚类，聚类不需要进行预测

print('k均值模型:\n',clf)
print('聚类结果:\n',y_pred)

X = origin.values
x1 = [n[0] for n in X]
x2 = [n[1] for n in X]
x3 = [n[2] for n in X]

plt.scatter(x1, x2, c=y_pred, marker='x')
plt.title("Kmeans")

plt.xlabel("Age")
plt.ylabel("Na")
plt.show()

