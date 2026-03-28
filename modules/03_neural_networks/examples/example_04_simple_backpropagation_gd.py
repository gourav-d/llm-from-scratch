import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feedforward(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backpropagation(X, y, z1, a1, z2, a2, w1, w2, b1, b2):
    dz2 = (a2 - y) * sigmoid_derivative(z2)
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    dz1 = np.dot(dz2, w2.T) * sigmoid_derivative(z1)
    dw1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    return w1, b1, w2, b2

# define input and output
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# define weights and biases
w1 = np.array([[3, 3],[-4, -4]])
b1 = np.array([2, -2])
w2 = np.array([[-4],[4]])
b2 = np.array([2])

learning_rate = 0.1

# train the network
for i in range(1000):
    z1, a1, z2, a2 = feedforward(X, w1, b1, w2, b2)
    w1, b1, w2, b2 = backpropagation(X, y, z1, a1, z2, a2, w1, w2, b1, b2)

# print the final output
z1, a1, z2, a2 = feedforward(X, w1, b1, w2, b2)
print(a2)
