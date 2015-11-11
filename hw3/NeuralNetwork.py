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

    def update_stochastic(self, x, y, lrate):
        # Propagate (i.e. feed-forward pass)

        fh = self.hiddenL.apply(x)
        ho = self.outputL.apply(fh)
        # Backpropagate (and update layers)
        y_input = np.zeros(self.class_dim)
        y_input[y] = 1.0
        y_input = (ho - y_input) / (ho * (1 - ho))
        dh = self.outputL.backprop_stochastic(fh, lrate, y_input, self.lam)
        dx = self.hiddenL.backprop_stochastic(x, lrate, dh, self.lam)
        return np.dot(np.log(ho), y_input)

    def update_batch(self, X, Y, lrate):
        fh = self.hiddenL.apply(X)
        ho = self.outputL.apply(fh)
        y_input = np.zeros((X.shape[0], self.class_dim))
        for i, y in enumerate(Y[:, 0]):
            y_input[i, y] = 1.0
        y_input = (ho - y_input) / (ho * (1 - ho))
        dh = self.outputL.backprop_batch(fh, lrate, y_input, self.lam)
        dx = self.hiddenL.backprop_batch(X, lrate, dh, self.lam)


class NNLayer(object):
    def __init__(self, idim, odim):
        self.idim = idim
        self.odim = odim
        #self.W = np.ones((self.idim, self.odim))
        self.W = np.random.rand(self.idim, self.odim) * 0.6 - 0.3
        self.Wo = np.zeros(self.odim,)
        # adaGrad sum squared gradients
        self.G2o = 1e-12 * np.ones((self.odim,))
        self.G2 = 1e-12 * np.ones((self.odim,))
        self.f = np.zeros((self.odim,))  # activation of output units

    def apply(self, x):
        self.f = expit(self.Wo + np.dot(x, self.W))
        return self.f

    def backprop_stochastic(self, x, lrate, delta, lam):
        grad = (1.0 - self.f) * self.f * delta  # dJ/dz = df/dz * delta.  (dtanh/dx = 1 - tanh^2)
        xdelta = np.dot(self.W, grad)        # dJ/dx to be returned
        xnorm2 = np.sum(x ** 2)
        self.G2o += grad ** 2
        self.G2 += xnorm2 * grad ** 2
        #self.Wo -= lrate * (grad + 2 * lam * self.Wo) / np.sqrt(self.G2o)
        self.Wo -= lrate * (grad) / np.sqrt(self.G2o)
        #self.W -= lrate * (np.outer(x, grad) + 2 * lam * self.W) / np.sqrt(self.G2)
        self.W -= lrate * (np.outer(x, grad)) / np.sqrt(self.G2)
        return xdelta

    def cpeff_(self):
        return self.W, self.Wo

    def backprop_batch(self, x, lrate, delta, lam):
        grad = (1 - self.f) * self.f * delta
        xdelta = np.dota(self.W, grad)
        com_grad = np.outer(x, grad)
        return com_grad, xdelta
