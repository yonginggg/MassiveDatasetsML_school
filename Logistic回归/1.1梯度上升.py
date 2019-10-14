import pandas as pd
import numpy as np
#导入csv数据
#dtype = str,最好读取的时候都以字符串的形式读入，不然可能会使数据失真
#比如一个0010008的编号可能会读取成10008

fileNameStr = 'D:/Work/数据挖掘ML/Logistic回归/telco.csv'

# encoding = "ISO-8859-1" -- 用什么解码，一般会默认系统的编码，如果是中文就用 "utf-8"
DataDF = pd.read_csv(fileNameStr,encoding = "utf-8")
DataDF = DataDF.fillna(method='ffill')
DataDF = DataDF.fillna(method='bfill')
DataDF = DataDF.values
# print(DataDF.shape)
DataX = DataDF[:,:-1]
DataX = np.hstack([np.ones((len(DataX), 1)) , DataX])
DataY = DataDF[:,-1]

# print(DataX.shape)
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
 
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
 
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)            
    labelMat = np.mat(classLabels).transpose() 
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):              
        h = sigmoid(dataMatrix*weights)     
        error = (labelMat - h)              
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights

w = gradAscent(DataX,DataY)
print("梯度上升得到的值为:")
print(w)

# 验证命中率
w = np.mat(w)
DataX = np.mat(DataX)
DataY = np.mat(DataY).T
loss = abs(DataY-sigmoid(DataX.dot(w)))
# print(type(loss))
# print(DataX.dot(w).shape)
# print(loss)
# print(DataY.shape)
count = 1
for i in range(1000):
    if loss[i]-0<0.1:
        count+=1


# print(count/1000)
# 0.757

