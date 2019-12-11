import pandas as pd
import numpy as np

X = pd.read_csv("X_data.csv",header=None)
y = pd.read_csv("y_label.csv",header=None)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#计算Sigmoid函数的偏导数
def sigmoid_derivative(x):
    return x * (1 - x)

def predictY(X, theta1, theta2):
    # theta1
    X = np.hstack([np.ones((len(X), 1)), X])
    k = X.dot(theta1.T)
    k = sigmoid(k)

    # theta2
    k = np.hstack([np.ones((len(k), 1)), k])
    k = k.dot(theta2.T)

    index = np.argmax(k, axis=1)
    index = index + 1

    return index

np.random.seed(1)
theta1 = np.random.randn(25,401)
theta2 = np.random.randn(10, 26)

result = predictY(X,theta1, theta2)

y_2 = []
for i in range(len(y)):
    y_2.append(y[0][i])

y = y_2
y = np.array(y)

error = y-result

adjustments = np.dot(X.T, error * sigmoid_derivative(result))

# print(adjustments)

theta11 = np.random.randn(25,401)+0.001
theta22 = np.random.randn(10, 26)+0.001

result2 = predictY(X,theta11, theta22)

error2 = y-result2

adjustments2 = np.dot(X.T, error2 * sigmoid_derivative(result2))

print(adjustments-adjustments2)