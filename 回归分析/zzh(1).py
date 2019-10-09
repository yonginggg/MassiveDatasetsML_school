import numpy as np
A = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_A.csv","rb"),delimiter=",",skiprows=0) 
B = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_B.csv","rb"),delimiter=",",skiprows=0) 
X_b =np.hstack([np.ones((len(A),1)),A])#插入一列全是1的列
theta =np.zeros(X_b.shape[1])#假定全为零的矩阵theta
def J(theta,X_b,y):#误差判断函数
    try:
        return np.sum(y-X_b.dot(theta)**2)/len(X_b)#误差计算
    except:
        return float('inf')#异常处理

def zzdj(theta,X_b,y,n,a,e,ren):
    assert A.shape[0] == B.shape[0]
    i=0
    for i in range(n):
         g=X_b.T.dot(X_b.dot(theta)-y)*2./len(y)#梯度下降，即J（theta）的导数
         l_theta=theta#保存上一个theta
         theta =theta*(1-a*(ren/len(y)))-a*g#新的theta，并进行正则化处理
         if(abs(J(theta,X_b,y)-J(l_theta,X_b,y))<e):#当误差判断小于给定值时
             return  i,theta
         i=i+1
    return "迭代次数达到上限 或者 学习率参数不准确"

print(zzdj(theta,X_b,B,10000,0.001,0.0000001,5))