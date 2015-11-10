#!/usr/bin/env python
from math import exp
import numpy as np
from plotBoundary import *
from cvxopt import matrix, solvers


__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def svm_train(X, Y, C, kernel):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Training set X, Y have different datapoints.")
    X = np.array(X)
    Y = np.array(Y)
    n = X.shape[0]

    # define your matrices
    # TODO: P needs to be rewritten for kernel function
    K = np.zeros((n, n))
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i, :], X[j, :])
            P[i, j] = Y[i, 0] * K[i, j] * Y[j, 0]
    P = matrix(P)
    q = -matrix(1.0, (n, 1))
    G = matrix(np.vstack((np.identity(n), -np.identity(n))))
    h = matrix(np.hstack((np.ones(n) * C, np.zeros(n))).T)
    A = matrix(np.squeeze(Y) * np.ones(n), (1, n))
    #A = matrix(Y.T)
    b = matrix(0.0, (1, 1))
    # find the solution
    solution = solvers.qp(P, q, G, h, A, b)
    alpha_vals = np.array(solution['x'])
    #theta = np.sum(alpha_vals * Y * X, axis=0)
    M = []
    for i in range(alpha_vals.shape[0]):
        if 0 < alpha_vals[i, 0] < C:
            M.append(i)
    theta_0 = 0.0
    for j in M:
        theta_0 += Y[j, 0] - np.dot(alpha_vals[:, 0].T * Y[:, 0].T, K[:, j])

    theta_0 = theta_0 / float(len(M))

    return (alpha_vals, theta_0)

"""
def predictSVM(x):
    return np.dot(theta, x) + theta_0
"""
if __name__ == '__main__':
    X = np.array([[1, 2],
                  [2, 2],
                  [0, 0],
                  [-2, 3]])
    Y = np.array([[1.0],
                  [1.0],
                  [-1.0],
                  [-1.0]])
    #theta, theta_0 = svm_train(X, Y, 1.0)
    #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'Test Example')

