#!/usr/bin/env python
from hw1.homework1 import regressTrainData, regressValidateData, regressTestData
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def sse_with_reg(X, Y, M, w, lam):
    ss = 0
    for i in range(len(Y)):
        approx_y = 0
        for j in range(M):
            approx_y += w[j] * (X[i] ** j)
        ss += abs(Y[i] - approx_y)
    ss += lam * (np.linalg.norm(w) ** 2)
    return ss


def sse(X, Y, w, M):
    ss = 0
    for i in range(len(Y)):
        approx_y = 0
        for j in range(M):
            approx_y += w[j] * (X[i] ** j)
        ss += abs(Y[i] - approx_y)
    return ss


def w_ml(X_train, Y_train, lam, M):
    w0 = np.zeros(M)

    def sse_with_reg_inside(w):
        ss = 0
        for i in range(len(Y_train)):
            approx_y = 0
            for j in range(M):
                approx_y += w[j] * (X_train[i] ** j)
            ss += abs(Y_train[i] - approx_y)
        ss += lam * (np.linalg.norm(w) ** 2)
        return ss

    ww = fmin_bfgs(sse_with_reg_inside, w0)
    return ww


def find_lam(X_train, Y_train, M, lam0, e, a):
    X_val_pre, Y_val_pre = regressValidateData()
    X_val = np.array(X_val_pre)[:, 0]
    Y_val = np.array(Y_val_pre)[:, 0]
    dx = 0.01
    while True:
        w_plus = w_ml(X_train, Y_train, lam0 + dx, M)
        w_minus = w_ml(X_train, Y_train, lam0 - dx, M)
        lam1 = lam0 - a * (sse(X_val, Y_val, w_plus, M) -
                           sse(X_val, Y_val, w_minus, M)) / (2 * dx)
        if abs(lam1 - lam0) < e:
            break
        else:
            lam0 = lam1
    w = w_ml(X_train, Y_train, lam1, M)
    ss = sse(X_val, Y_val, w, M)
    return lam1, w, ss

if __name__ == '__main__':
    """
    X_train, Y_train = regressTrainData()

    M = np.arange(1, 10)
    results = {}
    for m in M:
        lam, w, ss = find_lam(X_train, Y_train, m, 0.5, 0.01, 0.1)
        print m, lam, ss
        results[m] = {'w': w,
                      'lam': lam,
                      'sse': ss}
    for key, value in results.iteritems():
        print "{}: {}".format(key, value)
    """
    M = 2
    w = [1.12338195, 0.83457492]

    X_train, Y_train = regressTrainData()
    X_test, Y_test = regressTestData()
    X_val, Y_val = regressValidateData()

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
