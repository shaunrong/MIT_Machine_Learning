#!/usr/bin/env python
import scipy.io
import numpy as np
from hw3.multi_svm_train import multi_svm_train
import pylab as pl
from hw3.plotMultiDecisionBoundary import plotMultiDecisionBoundary

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

train = scipy.io.loadmat('data/mnist_train.csv')['tr']
X_train = train[:, 0:-1]
Y_train = train[:, -1:]
val = scipy.io.loadmat('data/mnist_validate.csv')['va']
X_val = val[:, 0:-1]
Y_val = val[:, -1:]
test = scipy.io.loadmat('data/mnist_test.csv')['te']
X_test = test[:, 0:-1]
Y_test = test[:, -1:]

print Y_train

"""

def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def poly_kernel(x1, x2):
    return (np.dot(x1, x2) + 1) ** 10


def predict_margin(x, X_train, Y_train, kernel, alpha, theta_0):
    n = X_train.shape[0]
    decision = 0
    for it in range(n):
        decision += alpha[it] * Y_train[it, 0] * kernel(x, X_train[it, :])
    decision += theta_0
    return decision
"""

"""
class ScoreFns(object):
    def __init__(self, X_train, Y_train, kernel, alpha, theta_0):
        self._X_train = X_train
        self._Y_train = Y_train
        self._kernel = kernel
        self._alpha = alpha
        self._theta_0 = theta_0

    def fn(self, x):
        #scoreFns = []
        def predict(x):
            n = X_train.shape[0]
            decision = 0
            for it in range(n):
                decision += self._alpha[it] * self._Y_train[it, 0] * self._kernel(x, X_train[it, :])
            decision += theta_0
            return decision
        return predict_margin(x, self._X_train, self._Y_train, self._kernel, self._alpha,
                              self._theta_0)

"""


def error_rate(X, Y, X_train, Y_train, alpha, theta_0, kernal):
    error = 0
    classes = np.sort(np.unique(Y_train), axis=None)
    for i in range(X.shape[0]):
        margin = {}
        for c in classes:
            Y_margin = np.array(Y_train==c, dtype=float)
            Y_margin = Y_margin * 2 - 1.0
            margin[c] = predict_margin(X[i, :], X_train, Y_margin, kernal, alpha[c], theta_0[c])
            predict_c = max(margin, key=margin.get)
        if margin[predict_c] < 0:
            error += 1
        elif predict_c != Y[i]:
            error += 1
    return float(error) / X.shape[0]
"""
#Parameters here
c = 100
kernel = poly_kernel
#Parameters here
alpha, theta, theta_0 = multi_svm_train(X_train, Y_train, c, kernel)

er_train = error_rate(X_train, Y_train, X_train, Y_train, alpha, theta_0, kernel)
print "er_train is {}".format(er_train)
er_val = error_rate(X_val, Y_val, X_train, Y_train, alpha, theta_0, kernel)
print "er_val is {}".format(er_val)
er_test = error_rate(X_test, Y_test, X_train, Y_train, alpha, theta_0, kernel)
print "er_test is {}".format(er_test)
"""
"""
scoreFns = []
for c in alpha.keys():
    score = ScoreFns(X_train, Y_train, kernal, alpha[c], theta_0[c])
    scoreFns.append(score.fn)

plotMultiDecisionBoundary(X_train, Y_train, scoreFns, [0, 0, 0], title="Linear Kernal, toy_1")
"""
#c = 1

#alpha, theta, theta_0 = multi_svm_train(X_train, Y_train, c, poly_kernel)
#er = error_rate(X_val, Y_val, X_train, Y_train, alpha, theta_0, poly_kernel)
#print er

"""
#for c in [1e3]:
c_list = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
er_list = []

for c in c_list:
    alpha, theta, theta_0 = multi_svm_train(X_train, Y_train, c, linear_kernel)
    er = error_rate(X_val, Y_val, X_train, Y_train, alpha, theta_0, linear_kernel)
    er_list.append(er)
    print c, er

c_log = np.log(c_list)

pl.plot(c_log, er_list, 'g^-')
pl.title('Model Selection on Linear Kernal, toy_multiclass_1')
pl.xlabel('Log(C)')
pl.ylabel('Error Rate on Validate Dataset')
pl.ylim([0.0, 1.2])
pl.show()



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
"""