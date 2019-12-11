#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/11/19 19:21
# @Author  : YangYusheng
# @File    : k均值.py
# @Software: PyCharm

from numpy import *
import matplotlib.pyplot as plt
import pandas as pd


# 加载本地数据
def loadDataSet(fileName):
    origin = pd.read_csv(fileName)
    origin = pd.get_dummies(origin)
    dataMat = origin.values
    return dataMat


# 欧式距离计算
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 格式相同的两个向量做运算


# 中心点生成 随机生成最小到最大值之间的值
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # 创建中心点，由于需要与数据向量做运算，所以每个中心点与数据得格式应该一致（特征列）
    for j in range(n):  # 循环所有特征列，获得每个中心点该列的随机值
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 获得每列的随机值 一列一列生成
    return centroids


# 返回 中心点矩阵和聚类信息
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 创建一个矩阵用于记录该样本 （所属中心点 与该点距离）
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False  # 如果没有点更新则为退出
        for i in range(m):
            minDist = inf;
            minIndex = -1
            for j in range(k):  # 每个样本点需要与 所有 的中心点作比较
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 距离计算
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 若记录矩阵的i样本的所属中心点更新，则为True，while下次继续循环更新
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2  # 记录该点的两个信息
        # print(centroids)
        for cent in range(k):  # 重新计算中心点
            # print(dataSet[nonzero(clusterAssment[:,0] == cent)[0]]) # nonzero返回True样本的下标
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 得到属于该中心点的所有样本数据
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 求每列的均值替换原来的中心点
    return centroids, clusterAssment


X = loadDataSet('DRUG1n.csv')
datMat = mat(loadDataSet('DRUG1n.csv'))
myCentroids, clustAssing = kMeans(datMat, 5)
print(myCentroids)
print(clustAssing)


a = array(clustAssing)
b = []
for i in range(a.shape[0]):
    b.append(a[i][0])
b = array(b)

x1 = [n[0] for n in X]
x2 = [n[1] for n in X]
x3 = [n[2] for n in X]
# 绘制散点图 参数：x横轴 y纵轴 c=y_pred聚类预测结果 marker类型 o表示圆点 *表示星型 x表示点
plt.scatter(x1, x2, c=b, marker='x')
plt.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],color='r',s=60)
# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("Age")
plt.ylabel("Na")

# 显示图形
plt.show()