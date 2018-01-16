import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process import get_data
def softmax(R):
    E=np.exp(R)
    return E/E.sum(axis=0, keepdims=True)

def forward(X,W,b,V,c):
    Z=1/(1+np.exp(-X.dot(W)-b))
    R=Z.dot(V)+c
    return softmax(R)

Xtrain, Ytrain, Xtest, Ytest=get_data()
D=Xtrain.shape[1]
M=5
K=len(set(Ytrain))
W=np.random.randn(D,M)
b=np.zeros(M)
V=np.random.randn(M,K)
c=np.zeros(K)
pred=forward(Xtrain,W,b,V,c);
predictions=np.argmax(pred,axis=1)

def classification_rate(Y,P):
    return np.mean(Y==P)

print("Score:", classification_rate(Ytrain, predictions))

Pred=forward
