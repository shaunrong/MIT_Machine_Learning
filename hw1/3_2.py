#!/usr/bin/env python
from hw1.homework1 import regressTrainData, regressValidateData, regressTestData
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def phi(X, M):
    X = np.array(X)
    N = X.shape[0]
    phi = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            phi[i, j] = X[i] ** j
    return phi


def w_ml(phi, Y, lam, M):
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi) + lam * np.identity(M)), phi.T)
    w = np.dot(w, Y)
    return w


def validate(M, lam):
    X_train, Y_train = regressTrainData()
    X_validate, Y_validate = regressValidateData()

    phi_train = phi(X_train, M)
    w = w_ml(phi_train, Y_train, lam, M)

    phi_validate = phi(X_validate, M)
    Y_approx = np.dot(phi_validate, w)
    sse = np.linalg.norm(Y_validate - Y_approx)

    return sse


def find_lam(M, lam0, e, a):
    iteration = 0
    dx = 0.00001
    while True:
        iteration += 1
        lam1 = lam0 - a * (validate(M, lam0 + dx) - validate(M, lam0 - dx)) / (2 * dx)
        if abs(lam1 - lam0) < e:
            break
        else:
            lam0 = lam1
    sse = validate(M, lam1)
    return iteration, lam1, sse

if __name__ == '__main__':
    """
    M = np.arange(1, 10)
    results = {}
    for m in M:
        it, lam, sse = find_lam(m, 0.5, 0.01, 0.1)
        print m, lam, sse
        results[m] = {'iteration': it,
                      'lam': lam,
                      'sse': sse}
    """
    M = 5
    lam = 0.81

    X_train, Y_train = regressTrainData()
    X_test, Y_test = regressTestData()
    X_val, Y_val = regressValidateData()

    phi_train = phi(X_train, M)
    w = w_ml(phi_train, Y_train, lam, M)
    phi_test = phi(X_test, M)
    Y_approx = np.dot(phi_test, w)
    sse = np.linalg.norm(Y_test - Y_approx) ** 2
    pts = [p for p in np.linspace(min(X_test), max(X_test), 100)]
    plt.plot(X_test.T.tolist()[0], Y_test.T.tolist()[0], 'gs')

    y_test_approx = []
    for pt in pts:
        s = 0
        for k in range(M):
            s += w[k] * (pt ** k)
        y_test_approx.append(s)

    plt.plot(pts, y_test_approx)
    plt.title('Test dataset')
    plt.show()

    plt.plot(X_train.T.tolist()[0], Y_train.T.tolist()[0], 'gs')
    y_test_approx = []
    for pt in pts:
        s = 0
        for k in range(M):
            s += w[k] * (pt ** k)
        y_test_approx.append(s)
    plt.plot(pts, y_test_approx)
    plt.title('Train dataset')
    plt.show()

    plt.plot(X_val.T.tolist()[0], Y_val.T.tolist()[0], 'gs')
    y_test_approx = []
    for pt in pts:
        s = 0
        for k in range(M):
            s += w[k] * (pt ** k)
        y_test_approx.append(s)
    plt.plot(pts, y_test_approx)
    plt.title('Validation dataset')
    plt.show()