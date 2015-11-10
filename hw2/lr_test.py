from math import log, exp
from plotBoundary import *
from scipy.optimize import fmin_bfgs
# import your LR training code
import numpy as np

# parameters
data = 'nonsep'
print '======Training======'
# load data from csv files
train = np.loadtxt('data/data_' + data + '_train.csv')
X = train[:, 0:2]
Y = train[:, 2:3]

print X.shape
print Y.shape
print Y[0, 0]

# Carry out training.
### TODO ###
lam = 0
def LR_object(w):
    nll = 0
    for i in range(Y.shape[0]):
        nll += log(1 + exp(-Y[i] * (np.dot(X[i], w[1:].T) + w[0])))
    nll += lam * np.dot(w, w)
    return nll

w_train = fmin_bfgs(LR_object, np.zeros(3))

print "w_train is {}".format(w_train)

# Define the predictLR(x) function, which uses trained parameters
### TODO ###
def predictLR(x):
    return 1.0 / (1.0 + exp(-(np.dot(x, w_train[1:]) + w_train[0])))

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = data + ' LR Train')

error_num = 0
for i in range(Y.shape[0]):
    predict = predictLR(X[i])
    if predict > 0.5:
        predict_y = 1.0
    else:
        predict_y = -1.0
    if abs(predict_y - Y[i]) > 0.01:
        error_num += 1

print "the error number is {}, error rate is {}".format(error_num, error_num/float(Y.shape[0]))

print '======Validation======'
# load data from csv files
validate = np.loadtxt('data/data_'+data+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

error_num = 0
for i in range(Y.shape[0]):
    predict = predictLR(X[i])
    if predict > 0.5:
        predict_y = 1.0
    else:
        predict_y = -1.0
    if abs(predict_y - Y[i]) > 0.01:
        error_num += 1

print "the error number is {}, error rate is {}".format(error_num, error_num/float(Y.shape[0]))

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = data + ' LR Validate')
