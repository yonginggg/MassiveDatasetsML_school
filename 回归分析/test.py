import numpy as np
x = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_A.csv","rb"),delimiter=",",skiprows=0) 
y = np.loadtxt(open("D:/Work/数据挖掘ML/回归分析/randn_data_regression_B.csv","rb"),delimiter=",",skiprows=0) 
x[:,0]=x[:,0]*1000

def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')

def dJ(theta, X_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
    
    return res * 2 / len(X_b) # 对于一个二维数组，len返回其行数


def dJ(theta, X_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
    
    return res * 2 / len(X_b) # 对于一个二维数组，len返回其行数
 
def gradient_descent(X_b, y, initial_theta, eta, n_iters = 100000, epsilon=1e-5, lan=500):  
    theta = initial_theta
    i_iter = 0
    
    while i_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta*(1-eta*(lan/len(y))) - eta*gradient
        
        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
            
        i_iter += 1
    print(i_iter)   
    return theta
 
X_b = np.hstack([np.ones((len(x), 1)) , x])
initial_theta = np.zeros(X_b.shape[1])  # 每个特征对应一个theta，（还应再多一个theta0）
eta = 0.01
theta = gradient_descent(X_b, y, initial_theta, eta)
 
# print(theta.shape)  # 结果对应截距和斜率
theta = np.matrix(theta)
print(X_b*theta.T)