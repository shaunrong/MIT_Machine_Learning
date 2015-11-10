#!/usr/bin/env python
import numpy as np
import scipy.io
from scipy.optimize import fmin_bfgs
from math import log, exp

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

train = scipy.io.loadmat('data/data_titanic_train.csv')['data']
X_train = train[:, 0:11]
Y_train = train[:, 11:12]
val = scipy.io.loadmat('data/data_titanic_validate.csv')['data']
X_val = val[:, 0:11]
Y_val = val[:, 11:12]
test = scipy.io.loadmat('data/data_titanic_test.csv')['data']
X_test = test[:, 0:11]
Y_test = test[:, 11:12]


def feature_scaling(X_train, X_val, X_test):
    for j in range(X_train.shape[1]):
        x_min_train = min(X_train[:, j])
        x_min_val = min(X_val[:, j])
        x_min_test = min(X_test[:, j])
        x_max_train = max(X_train[:, j])
        x_max_val = max(X_val[:, j])
        x_max_test = max(X_test[:, j])
        x_min = min(x_min_train, x_min_val, x_min_test)
        x_max = max(x_max_train, x_max_val, x_max_test)

        X_train[:, j] = (X_train[:, j] - x_min) / float(x_max - x_min)
        X_val[:, j] = (X_val[:, j] - x_min) / float(x_max - x_min)
        X_test[:, j] = (X_test[:, j] - x_min) / float(x_max - x_min)
    return X_train, X_val, X_test


X_train, X_val, X_test = feature_scaling(X_train, X_val, X_test)

global lam
lam = 0.0


def LR_object(w):
    global lam
    nll = 0
    for i in range(Y_train.shape[0]):
        nll += log(1 + exp(-Y_train[i] * (np.dot(X_train[i], w[1:].T) + w[0])))
    nll += lam * np.dot(w, w)
    return nll


def error_rate(X, Y, w):
    error = 0
    for i in range(X.shape[0]):
        if Y[i] * (np.dot(X[i], w[1:].T) + w[0]) < 0:
            error += 1
    return float(error) / X.shape[0]

"""
w_train = fmin_bfgs(LR_object, np.zeros(12))
val_error_rate = error_rate(X_train, Y_train, w_train)
print "error rate is {}".format(val_error_rate)

while True:
    lam += 0.01
    w_train = fmin_bfgs(LR_object, np.zeros(12))
    er = error_rate(X_val, Y_val, w_train)
    print "error rate is {}".format(er)
    if er < val_error_rate:
        val_error_rate = er
    else:
        break
"""
min_er = 1.0
best_l = 0.0

results = {}
for l in np.arange(0, 3, 0.1):
    w_train = fmin_bfgs(LR_object, np.zeros(12))
    er = error_rate(X_val, Y_val, w_train)
    results[l] = {'er': er, 'w': w_train}
    if er < min_er:
        best_l = l
        min_er = er


print "error rate on training set is {}".format(error_rate(X_train, Y_train, results[best_l]['w']))
print "error rate on validation set is {}".format(results[best_l]['er'])
print "lambda is {}".format(best_l)
print "error rate on test set is {}".format(error_rate(X_test, Y_test, results[best_l]['w']))
print "the training w is {}".format(results[best_l]['w'])
    





