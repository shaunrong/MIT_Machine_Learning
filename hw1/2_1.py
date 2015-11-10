#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

from homework1 import regressionPlot, bishopCurveData

if __name__ == '__main__':
    X, Y = bishopCurveData()
    regressionPlot(X, Y, 0)