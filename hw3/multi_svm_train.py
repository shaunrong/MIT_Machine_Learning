#!/usr/bin/env python
import numpy as np
from cvxopt import matrix, solvers


__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def multi_svm_train(X, Y, C, kernel):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Training set X, Y have different datapoints.")
    X = np.array(X)
    multi_Y = np.array(Y)
    classes = np.sort(np.unique(multi_Y), axis=None)
    n = X.shape[0]
    alpha_vals = {}
    theta = {}
    theta_0 = {}

    #Define some commonly used matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i, :], X[j, :])
    q = -matrix(1.0, (n, 1))
    G = matrix(np.vstack((np.identity(n), -np.identity(n))))
    h = matrix(np.hstack((np.ones(n) * C, np.zeros(n))).T)
    b = matrix(0.0, (1, 1))

    for c in classes:
        Y = np.array(multi_Y == c, dtype=float)
        # define your matrices
        Y = Y * 2 - 1.0
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = Y[i, 0] * K[i, j] * Y[j, 0]
        P = matrix(P)
        A = matrix(np.squeeze(Y) * np.ones(n), (1, n))
        # find the solution
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alpha_vals_c = np.array(solution['x'])
        theta_c = np.sum(alpha_vals_c * Y * X, axis=0)
        M = []
        for i in range(alpha_vals_c.shape[0]):
            if 0.0 < alpha_vals_c[i, 0] < C:
                M.append(i)
        theta_0_c = 0.0
        for j in M:
            theta_0_c += Y[j, 0] - np.dot(alpha_vals_c[:, 0].T * Y[:, 0].T, K[:, j])

        theta_0_c = theta_0_c / float(len(M))

        alpha_vals[c] = alpha_vals_c
        theta[c] = theta_c
        theta_0[c] = theta_0_c

    return alpha_vals, theta, theta_0





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

