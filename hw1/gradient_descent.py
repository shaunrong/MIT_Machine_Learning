#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

from scipy.optimize import fmin_bfgs
import numpy as np

def blow_gd(x1_0, x2_0, e, a):
    iteration = 0
    while True:
        iteration += 1
        x1_1 = x1_0 - 2 * a * (x1_0 - 1)
        x2_1 = x2_0 - 2 * a * (x2_0 - 2)
        if abs(x1_0 - x1_1) < e and abs(x2_0 - x2_1) < e:
            break
        else:
            x2_0 = x2_1
            x1_0 = x1_1
    return iteration, x1_1, x2_1


def double_mini_gd(x_0, e, a):
    iteration = 0
    while True:
        iteration += 1
        x_1 = x_0 - a * (4 * x_0**3 - 10 * x_0)
        if abs(x_0 -  x_1) < e:
            break
        else:
            x_0 = x_1
    return iteration, x_1


def blow_gd_numerical(x1_0, x2_0, e, a):
    iteration = 0
    dx = 0.00001
    while True:
        iteration += 1
        x1_1 = x1_0 - a * (blow_f(x1_0 + dx, x2_0) - blow_f(x1_0 - dx, x2_0)) / (2 * dx)
        x2_1 = x2_0 - a * (blow_f(x1_0, x2_0 + dx) - blow_f(x1_0, x2_0 - dx)) / (2 * dx)
        if abs(x1_0 - x1_1) < e and abs(x2_0 - x2_1) < e:
            break
        else:
            x2_0 = x2_1
            x1_0 = x1_1
    return iteration, x1_1, x2_1


def blow_f(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2


def double_mini_gd_numerical(x_0, e, a):
    iteration = 0
    dx = 0.00001
    while True:
        iteration += 1
        x_1 = x_0 - a * (double_mini(x_0 + dx) - double_mini(x_0 - dx)) / (2 * dx)
        if abs(x_0 - x_1) < e:
            break
        else:
            x_0 = x_1
    return iteration, x_1


def double_mini(x):
    return (x + 2) * (x + 1) * (x - 1) * (x - 2)

if __name__ == '__main__':
    print fmin_bfgs(blow_f, np.array([0.0, 0.0]))