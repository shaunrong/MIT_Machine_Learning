#!/usr/bin/env python
import numpy as np
from scipy.optimize import fmin_bfgs

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def phi(X):
    return X


def w_ml(phi, Y, lam):
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi) + lam * np.identity(phi.shape[1])), phi.T)
    w = np.dot(w, Y)
    return w

"""
def validate(X_train, Y_train, X_val, Y_val, lam):

    phi_train = phi(X_train)
    w = w_ml(phi_train, Y_train, lam)

    phi_validate = phi(X_val)
    Y_approx = np.dot(phi_validate, w)
    sse = np.linalg.norm(Y_val - Y_approx) ** 2

    return sse
"""

def find_lam(X_train, Y_train, X_test, Y_test, X_val, Y_val, lam0, e, a):
    iteration = 0
    dx = 0.00001
    while True:
        iteration += 1
        lam1 = lam0 - a * (validate(X_train, Y_train, X_val, Y_val, lam0 + dx) -
                           validate(X_train, Y_train, X_val, Y_val, lam0 - dx)) / (2 * dx)
        if abs(lam1 - lam0) < e:
            break
        else:
            lam0 = lam1
    sse_val = validate(X_train, Y_train, X_val, Y_val, lam1)
    sse_test = validate(X_train, Y_train, X_test, Y_test, lam1)
    return lam1, sse_val, sse_test


def feature_scaling(X_train, X_test, X_val):
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[1] != X_val.shape[1] or X_test.shape[1]!= X_val.shape[1]:
        raise ValueError('X_train, test, val has different dimensions')
    for j in range(X_train.shape[1]):
        x_min = min([min(X_train[:, j]), min(X_test[:, j]), min(X_val[:, j])])
        x_max = max([max(X_train[:, j]), max(X_test[:, j]), max(X_val[:, j])])
        if (x_max - x_min) != 0:
            X_train[:, j] = (X_train[:, j] - x_min) / (x_max - x_min)
            X_test[:, j] = (X_test[:, j] - x_min) / (x_max - x_min)
            X_val[:, j] = (X_val[:, j] - x_min) / (x_max - x_min)
    return X_train, X_test, X_val


if __name__ == '__main__':
    def validate(lam):

        phi_train = phi(X_train)
        w = w_ml(phi_train, Y_train, lam)

        phi_validate = phi(X_val)
        Y_approx = np.dot(phi_validate, w)
        sse = np.linalg.norm(Y_val - Y_approx) ** 2

        return sse

    Y_train = np.genfromtxt('BlogFeedback_data/y_train.csv', dtype=float, delimiter=',').T
    X_train = np.genfromtxt('BlogFeedback_data/x_train.csv', dtype=float, delimiter=',')
    X_test = np.genfromtxt('BlogFeedback_data/x_test.csv', dtype=float, delimiter=',')
    Y_test = np.genfromtxt('BlogFeedback_data/y_test.csv', dtype=float, delimiter=',').T
    X_val = np.genfromtxt('BlogFeedback_data/x_val.csv', dtype=float, delimiter=',')
    Y_val = np.genfromtxt('BlogFeedback_data/y_val.csv', dtype=float, delimiter=',').T
    X_train, X_test, X_val = feature_scaling(X_train, X_test, X_val)

    lam = fmin_bfgs(validate, np.array([0.1]))

    print lam

    phi_train = phi(X_train)
    w = w_ml(phi_train, Y_train, lam)

    phi_validate = phi(X_val)
    Y_approx = np.dot(phi_validate, w)
    sse_val = np.linalg.norm(Y_val - Y_approx) ** 2 / Y_val.shape[0]

    phi_test = phi(X_test)
    Y_approx_test = np.dot(phi_test, w)
    sse_test = np.linalg.norm(Y_test - Y_approx) ** 2 / Y_test.shape[0]

    print sse_test, sse_val




    """
    M = 5
    lam = 0.81

    X_train, Y_train = regressTrainData()
    X_test, Y_test = regressTestData()

    phi_train = phi(X_train, M)
    w = w_ml(phi_train, Y_train, lam, M)
    phi_test = phi(X_test, M)
    Y_approx = np.dot(phi_test, w)
    sse = np.linalg.norm(Y_test - Y_approx) ** 2
    print sse
    """