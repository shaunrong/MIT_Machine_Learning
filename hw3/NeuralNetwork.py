#!/usr/bin/env python
import numpy as np
from scipy.special import expit


__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


class NeuralNet(object):
    def __init__(self, idim, hdim, class_dim, lam):
        self.idim = idim
        self.hdim = hdim      # number of hidden layer units
        self.class_dim = class_dim    # vocab size

        self.hiddenL = NNLayer(self.idim, self.hdim)
        self.outputL = NNLayer(self.hdim, self.class_dim)
        self.lam = lam

    def predict(self, x):
        fh = self.hiddenL.apply(x)
        proba = self.outputL.apply(fh)
        proba = list(proba)
        return proba.index(max(proba))

    def update(self, x, y, lrate):
        # Propagate (i.e. feed-forward pass)

        fh = self.hiddenL.apply(x)
        ho = self.outputL.apply(fh)
        # Backpropagate (and update layers)
        y_input = np.ones(self.class_dim) * (-1.0)
        y_input[y] = 1.0
        y_input = - 1 * y_input / ho
        dh = self.outputL.backprop(fh, lrate, y_input, self.lam)
        dx = self.hiddenL.backprop(x, lrate, dh, self.lam)
        return np.dot(np.log(ho), -y_input)


class NNLayer(object):
    def __init__(self, idim, odim):
        self.idim = idim
        self.odim = odim
        self.W = np.random.rand(self.idim, self.odim)
        self.Wo = np.zeros(self.odim,)
        # adaGrad sum squared gradients
        self.G2o = 1e-12 * np.ones((self.odim,))
        self.G2 = 1e-12 * np.ones((self.odim,))
        self.f = np.zeros((self.odim,))  # activation of output units

    def apply(self, x):
        self.f = expit(self.Wo + np.dot(x, self.W))
        return self.f

    def backprop(self, x, lrate, delta, lam):
        grad = (1.0 - self.f) * self.f * delta  # dJ/dz = df/dz * delta.  (dtanh/dx = 1 - tanh^2)
        xdelta = np.dot(self.W, grad)        # dJ/dx to be returned
        xnorm2 = np.sum(x ** 2)
        self.G2o += grad ** 2
        self.G2 += xnorm2 * grad ** 2
        self.Wo -= lrate * grad / np.sqrt(self.G2o)
        self.W -= lrate * np.outer(x, grad / np.sqrt(self.G2))
        return xdelta
