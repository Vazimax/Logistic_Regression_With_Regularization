import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

path = r"C:\Users\aboub\OneDrive\Desktop\ML_class\ex2\ex2data2.txt"

data = pd.read_csv(path, header=None, 
                   names=['Test 1', 'Test 2', 'Accepted'])

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(positive['Test 1'], positive['Test 2'],
           s=50, c='g', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'],
           s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test1 Score')
ax.set_ylabel('Test2 Score')

degree = 5

x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j) # i=3 , j=2

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Cost_Reg(theta, X, y, lr ):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (lr / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    
     
    return np.sum(first - second) / (len(X)) + reg


def Gradient_Reg(theta, X, y, alpha):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] =(np.sum(term)/len(X))+((alpha/len(X))*theta[:,i])
    
    return grad

cols = data.shape[1]

X2 = data.iloc[:,1:cols]
y2 = data.iloc[:,0:1]

X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(X2.shape[1])

alpha = 0.00001

rcost = Cost_Reg(theta2, X2, y2, alpha)

result = opt.fmin_tnc(func=Cost_Reg, x0=theta2, fprime=Gradient_Reg, args=(X2, y2, learningRate))

print(f'result = {result} \n')

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print (f'accuracy = {accuracy}%')
