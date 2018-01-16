import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# so scripts from other folders can import this file
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def get_data():
    dataset = pd.read_csv("ecommerce_data.csv")
    df = dataset.as_matrix()

    X = df[:, :-1]
    Y = df[:, -1].astype(np.int32)

    N,D=X.shape
    X2=np.zeros((N,D+3))
    X2[:,0:(D-1)]=X[:,0:(D-1)]
    for i in range(N):
        t=int(X[i,D-1])
        X2[i,t+D-1]=1
    X=X2
    Xtrain=X[:-100]
    Xtest=X[-100:]
    Ytrain=Y[:-100]
    Ytest=Y[-100:]
    for i in (1,2):
        m=Xtrain[:,i].mean()
        s=Xtrain[:,i].std()
        Xtrain[:,i]=(Xtrain[:,i]-m)/s
        Xtest[:,i]=(Xtest[:,i]-m)/s
    return Xtrain, Ytrain, Xtest, Ytest

def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest=get_data()
    X2train=Xtrain[Ytrain<=1]
    Y2train=Ytrain[Ytrain<=1]
    X2test=Xtest[Ytrain<=1]
    Y2test=Ytest[Ytrain<=1]
    return X2train, Y2train, X2test, Y2test





