#!/usr/bin/env python
import numpy as np
import scipy.io
from scipy.optimize import fmin_bfgs
from math import log, exp
from hw2.svm_train import svm_train

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


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def error_rate(X, Y, theta, theta_0):
    error = 0
    for i in range(X.shape[0]):
        if Y[i] * (np.dot(X[i], theta) + theta_0) < 0:
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
best_c = 0.1
results = {}

results = {}
for c in np.arange(0.1, 1, 0.1):
    print Y_train.shape
    alpha, theta_0 = svm_train(X_train, Y_train, c, linear_kernel)
    theta = np.sum(alpha * Y_train * X_train, axis=0)
    er = error_rate(X_val, Y_val, theta, theta_0)
    if er < min_er:
        min_er = er
        best_c = c
    results[c] = {"theta": theta, "er": er, 'theta_0': theta_0}


print "error rate on training set is {}".format(error_rate(X_train, Y_train, results[best_c]['theta'],
                                                           results[best_c]['theta_0']))
print "error rate on validation set is {}".format(results[best_c]['er'])
print "lambda is {}".format(best_c)
print "error rate on test set is {}".format(error_rate(X_test, Y_test, results[best_c]['theta'],
                                                       results[best_c]['theta_0']))
print "the training w is {}".format(results[best_c]['theta'])
