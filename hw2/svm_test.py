from numpy import *
from plotBoundary import *
from svm_train import svm_train
from math import exp
# import your SVM training code

# parameters
name = 'nonsep'
C = 100
sigma = 1
print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()


# Carry out training, primal and/or dual
def gaussian_kernel(x1, x2):
    return exp(-np.dot(x1-x2, x1-x2) / (2 * sigma * sigma))

alpha, theta_0 = svm_train(X, Y, C, gaussian_kernel)

theta = np.sum(alpha * Y * X, axis=0)

print "1/|w| is {}".format(1.0 / np.linalg.norm(theta))

support_vector_num = 0
for a in alpha:
    if abs(a) < 1e-6:
        support_vector_num += 1

print "number of support vector is {}".format(support_vector_num)

# Define the predictSVM(x) function, which uses trained parameters
"""
def predictSVM(x):
    return np.dot(theta, x) + theta_0
"""


def predictSVM(x):
    n = X.shape[0]
    decision = 0
    for it in range(n):
        decision += alpha[it] * Y[it, 0] * gaussian_kernel(x, X[it, :])
    decision += theta_0
    return decision


# plot training results
#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = name + ' SVM Train')

error_num = 0
for i in range(Y.shape[0]):
    if predictSVM(X[i, :]) * Y[i] < 0:
        error_num += 1
print "error number is {}, error rate is {}.".format(error_num, error_num / float(Y.shape[0]))


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = name + ' SVM Validate')

error_num = 0
for i in range(Y.shape[0]):
    if predictSVM(X[i, :]) * Y[i] < 0:
        error_num += 1
print "error number is {}, error rate is {}.".format(error_num, error_num / float(Y.shape[0]))


