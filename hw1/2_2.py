#!/usr/bin/env python
from hw1.homework1 import bishopCurveData

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def gd(f, x, dx):
    return (f(x + dx) - f(x - dx)) / (2 * dx)


def sse(w):
    X, Y = bishopCurveData()
    sse = 0
    for i in range(len(Y)):
        sse += (Y[i] - w) ** 2
    return sse

if __name__ == '__main__':
    print gd(sse, 0, 0.00001)
    X, Y = bishopCurveData()
    v = 0
    for y in Y:
        v += 2 * y
    print v