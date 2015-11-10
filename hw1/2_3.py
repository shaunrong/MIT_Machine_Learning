#!/usr/bin/env python
from hw1.homework1 import bishopCurveData, designMatrix
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np


def grad(f, x, dx):
    order = x.shape[0]
    df = np.zeros(order)
    for dim in range(order):
        add_vec = np.zeros(order)
        add_vec[dim] = 1
        add_vec *= dx
        df[dim] = (f(x + add_vec) - f(x - add_vec)) / (2 * dx)
    return df


def sse(w):
    w = np.array(w)
    order = w.shape[0]
    X, Y = bishopCurveData()

    ss = 0
    for i in range(len(Y)):
        approx = 0
        for j in range(order):
            approx += w[j] * (X[i]**j)
        ss += (Y[i] - approx) ** 2

    return ss


def gd_sse(w0, e, a):
    w0 = np.array(w0)
    iteration = 0
    dx = 0.00001
    while True:
        iteration += 1
        gd = grad(sse, w0, dx)
        w1 = w0 - a * gd
        converge = True
        for i in range(w1.shape[0]):
            if abs(w1[i] - w0[i]) > e:
                converge = False
        if converge:
            break
        else:
            w0 = w1
    return iteration, w1


if __name__ == '__main__':
    order = 10
    w0 = np.zeros(order)
    w1 = fmin_bfgs(sse, w0)
    #it, w1 = gd_sse(w0, 0.0001, 0.01)
    X, Y = bishopCurveData()

    pts = [p for p in np.linspace(min(X), max(X), 100)]
    plt.plot(X.T.tolist()[0], Y.T.tolist()[0], 'gs')

    y_approx = []
    for pt in pts:
        s = 0
        for k in range(order):
            s += w1[k] * (pt ** k)
        y_approx.append(s)

    plt.plot(pts, y_approx)
    plt.title('Fit Order M = {}'.format(order))
    plt.show()
