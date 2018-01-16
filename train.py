import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from process import get_data

Xtrain, Ytrain, Xtest, Ytest=get_data()

def ind(Y,k):
    N=len(Y)
    s=np.zeros((N,k))
    for i in range(N):
        s[i,Y[i]]=1
    return s

D=Xtrain.shape[1]
k=len(set(Ytrain) | set(Ytest))
M=5
W=np.random.randn(D,M)
b1=np.zeros(M)
V=np.random.randn(M,k)
b2=np.zeros(k)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))
Ytrainind=ind(Ytrain,k)
Ytestind=ind(Ytest,k)
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain, Ztrain = forward(Xtrain, W, b1, V, b2)
    pYtest, Ztest = forward(Xtest, W, b1, V, b2)

    ctrain = cross_entropy(Ytrainind, pYtrain)
    ctest = cross_entropy(Ytestind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    V-= learning_rate*Ztrain.T.dot(pYtrain-Ytrainind)
    b2-= learning_rate*(pYtrain-Ytrainind).sum(axis=0)
    dz=(pYtrain - Ytrainind).dot(V.T)*(1-Ztrain*Ztrain)
    W-=learning_rate*Xtrain.T.dot(dz)
    b1-=learning_rate*dz.sum(axis=0)
    if(i%1000==0):
        print(i,ctrain,ctest)
print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()






