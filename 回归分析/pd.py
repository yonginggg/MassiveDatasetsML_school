import numpy as np
from sklearn.preprocessing import normalize

A = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_A.csv","rb"),delimiter=",",skiprows=0) 
B = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_B.csv","rb"),delimiter=",",skiprows=0) 

A[:,0]=A[:,0]*100000#第一行乘以100000
X_b =np.hstack([np.ones((len(A),1)),A])#插入一列全是1的列
X_b = normalize(X_b, axis=0, norm='max')#归一化处理
theta =np.zeros(X_b.shape[1])#假定全为零的矩阵theta
def J(theta,X_b,y):#误差判断函数
    try:
        
        return np.sum(y-X_b.dot(theta)**2)/len(X_b)#误差计算
    except:
        return float('inf')#异常处理

def dj(theta,X_b,y,n,a,e):
    assert A.shape[0] == B.shape[0]
    i=0
    for i in range(n):
         g=X_b.T.dot(X_b.dot(theta)-y)#梯度下降，即J（theta）的导数
         l_theta=theta#保存上一个theta
         theta =theta-a*g#新的theta
         if(abs(J(theta,X_b,y)-J(l_theta,X_b,y))<e):#当误差判断小于给定值时
             return  i,theta
         i=i+1
    return "迭代次数达到上限 或者 学习率参数不准确"

print(dj(theta,X_b,B,100000 ,0.001,0.00001))