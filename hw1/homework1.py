import pdb
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
import numpy as np
from math import sin, pi


def designMatrix(X, order):
    X = np.array(X)
    N = X.shape[0]
    phi = np.zeros((N, order))
    for i in range(N):
        for j in range(order):
            phi[i, j] = X[i] ** j
    return phi


def design_matrix_sin(X, order):
    X = np.array(X)
    N = X.shape[0]
    phi = np.zeros((N, order))
    for i in range(N):
        for j in range(order):
            if j == 0:
                phi[i, j] = 1
            else:
                phi[i, j] = sin(2 * pi * j * X[i])
    return phi


def regressionFit(X, Y, phi):
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
    w = np.dot(w, Y)
    return w


# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    plt.plot(X.T.tolist()[0], Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = design_matrix_sin(X, order)

    #print phi
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    print 'w', w
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, design_matrix_sin(pts, order).T)
    plt.plot(pts, Yp.tolist()[0])
    plt.title('Fit Order M = {}'.format(order))
    plt.show()


def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y


def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')


def regressTrainData():
    return getData('regress_train.txt')


def regressValidateData():
    return getData('regress_validate.txt')

def regressTestData():
    return getData('regress_test.txt')

if __name__ == '__main__':
    X, Y = bishopCurveData()
    regressionPlot(X, Y, 10)