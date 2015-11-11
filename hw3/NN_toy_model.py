#!/usr/bin/env python
import scipy.io
from NeuralNetwork import NeuralNet
import numpy as np
import matplotlib.pyplot as pl

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


train = scipy.io.loadmat('data/toy_multiclass_1_train.csv')['toy_data']
X_train = train[:, 0:-1]
Y_train = train[:, -1:] - 1
val = scipy.io.loadmat('data/toy_multiclass_1_validate.csv')['toy_data']
X_val = val[:, 0:-1]
Y_val = val[:, -1:] - 1
test = scipy.io.loadmat('data/toy_multiclass_1_test.csv')['toy_data']
X_test = test[:, 0:-1]
Y_test = test[:, -1:] - 1

lrate = 0.5
lam = 0.01
#hdims = np.arange(2, 20, 2)
hdims = [10]


def error_rate(X_val, Y_val, nn):
    error = 0
    for i in range(X_val.shape[0]):
        y = nn.predict(X_val[i, :])
        if y != Y_val[i, 0]:
            error += 1
    return error / float(X_val.shape[0])

er = []

for hdim in hdims:
    nn = NeuralNet(X_train.shape[1], hdim, 3, lam)
    for it in range(20):
        LL = 0
        for i in range(X_train.shape[0]):
            pr = nn.update_stochastic(X_train[i], Y_train[i, 0], lrate)
            LL += pr
            LL /= float(X_train.shape[0])

    er_val = error_rate(X_val, Y_val, nn)
    er_train = error_rate(X_train, Y_train, nn)
    er_test = error_rate(X_test, Y_test, nn)
    er.append(er_val)
    print "error rate on val db: hdim: {}, er rate: {}".format(hdim, er_val)
    print "error rate on train db: hdim: {}, er rate: {}".format(hdim, er_train)
    print "error rate on test db: hdim: {}, er rate: {}".format(hdim, er_test)



pl.plot(hdims, er, 'g^-')
pl.title('Model Selection on Neural Network, toy_multiclass_1')
pl.xlabel('Hidden Layer Units')
pl.ylabel('Error Rate on Validate Dataset')
pl.ylim([0.0, 0.02])
pl.show()
