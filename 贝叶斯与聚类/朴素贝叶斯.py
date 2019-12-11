import pandas as pd
import numpy as np
origin = pd.read_csv("german_clean.csv")
#独热
origin = pd.get_dummies(origin)
#移动class位置到最后列
cols = list(origin)
cols.insert(62,cols.pop(cols.index('class')))
origin = origin.loc[:,cols]
origin

#train test
originTrain = origin[:700]
originTest = origin[700:]
originTest = originTest.values
#提取class==1, ==2的行
origin1 = originTrain.loc[origin['class'].isin(['1'])]
origin2 = originTrain.loc[origin['class'].isin(['2'])]

origin1v = origin1.values
origin2v = origin2.values
len1 = len(origin1v)
len2 = len(origin2v)

p1 = len1/len(originTrain)
p2 = len2/len(originTrain)

def predict1(p):
    k=1
    for i in range (len(p)):
        k *= (origin1v[:,i].tolist().count(p[i])+1)/len1
    return k*p1

def predict2(p):
    b=1
    for i in range (len(p)):
        b *= (origin2v[:,i].tolist().count(p[i])+1)/len2
    return b*p2

X = originTest[:,:-1]
y = originTest[:,-1]

a = []
for i in range (len(X)):
    if predict1(X[i,:-1].tolist())-predict2(X[i,:-1].tolist())>0:
        a.append(1)
    else:
        a.append(2)

np.array(a)
f = y-a
count = 0
for i in range(len(f)):
    if abs(f[i]-0)<0.1:
        count+=1
print(count/len(f))