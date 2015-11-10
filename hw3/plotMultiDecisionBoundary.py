import pdb
import numpy as np
import pylab as pl

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot


def plotMultiDecisionBoundary(X, Y, scoreFns, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    classes = np.sort(np.unique(Y), axis=None)

    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    pl.figure()
    for i in range(len(scoreFns)):
        zz = np.array([scoreFns[i](x) for x in np.c_[xx.ravel(), yy.ravel()]])
        zz = zz.reshape(xx.shape)
        CS = pl.contour(xx, yy, zz, [values[i]], c=(1.0 - classes[i]), linestyles='solid', linewidths=2, cmap=pl.cm.cool)
        pl.clabel(CS, fontsize=9, inline=1)
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap=pl.cm.cool)
    pl.title(title)
    pl.axis('tight')
    pl.show()
    # Plot the training points

