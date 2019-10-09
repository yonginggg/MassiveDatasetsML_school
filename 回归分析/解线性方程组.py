import numpy as np

A = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_A.csv","rb"),delimiter=",",skiprows=0) 
B = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_B.csv","rb"),delimiter=",",skiprows=0) 
new_A = np.hstack([A,np.ones((len(A),1))])
A = np.matrix(A)
B = np.matrix(B)
new_A = np.matrix(new_A)

def linerRegression(X_train,Y_train):
    "伪逆矩阵,如果矩阵A是可逆（非奇异）的，那么pinv(A)与inv(A)的结果是一样的"
    theta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(Y_train.T))
    return theta

# print(new_A*linerRegression(new_A, B))
print(linerRegression(new_A,B))

