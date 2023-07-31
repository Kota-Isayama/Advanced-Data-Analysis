import numpy as np
import matplotlib
from scipy import linalg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)
def data_generation1(n=100):
    n = 100
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)],axis=1)

def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(
    np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)

def lpp(x, t, n_components=1):
    W = np.exp(-np.sum((x[:, None] - x[None]) ** 2, axis=2)/ (2. * t ** 2.))
    D = np.diag(np.sum(W, axis=1))
    L = D-W
    z = x.T.dot(D).dot(x)
    w, v = linalg.eig(x.T.dot(L).dot(x), z)
    return w[:n_components], v[:, :n_components]

n = 100
n_components = 1
# x = data_generation1(n)
x = data_generation2(n)
w, v = lpp(x, 1., n_components=n_components)
plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot(np.array([-v[0], v[0]]) * 9, np.array([-v[1], v[1]]) * 9)
plt.show()