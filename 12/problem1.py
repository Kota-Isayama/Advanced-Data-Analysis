import numpy as np
import matplotlib
from scipy import linalg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)

def data_generation1(n):
    x = np.random.randn(n, 2)
    x[0:n//2, 0] = x[0:n//2, 0] - 4
    x[n//2:, 0] = x[n//2:, 0] + 4
    x = x - np.mean(x, axis=0, keepdims=True)
    x[:, 1] = 5*x[:, 1]
    y = np.concatenate([np.zeros(n//2), np.ones(n - n//2)])
    return x, y

def data_generation2(n):
    x = np.random.randn(n, 2)
    x[0:n//4, 0] = x[0:n//4, 0] - 4
    x[n//4:n//2, 0] = x[n//4:n//2, 0] + 4
    x = x - np.mean(x, axis=0, keepdims=True)
    x[:, 1] = 5 * x[:, 1]
    y = np.concatenate([np.zeros(n//2), np.ones(n - n//2)])
    return x, y

def fda(x, y,  n_class, n_components=1):
    d = x.shape[1]
    Sb = np.zeros((d, d))
    for c in range(n_class):
        mu = np.mean(x[y==c], axis=0)
        n = np.sum(y==c)
        Sb += n * mu.dot(mu.T)
    C = np.zeros((d, d))
    for i in range(x.shape[0]):
        C += np.matmul(x[i].reshape(d, -1), x[i].reshape(-1, d))

    Sw = C - Sb

    w, v = linalg.eig(Sb, Sw)
    return w[:n_components], v[:, :n_components]

n = 100
n_components = 1
# x, y = data_generation1(n)
x, y = data_generation2(n)
w, v = fda(x, y, 2, n_components=n_components)
plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[y==0, 0], x[y==0, 1], 'bo')
plt.plot(x[y==1, 0], x[y==1, 1], 'rx')
plt.plot(np.array([-v[0], v[0]]) * 9, np.array([-v[1], v[1]]) * 9)
<<<<<<< HEAD
plt.savefig("/Users/kota/Documents/Graduate_School/M2_S/ADA/Advanced-Data-Analysis/12/prob1_3.png")
=======
plt.show()
>>>>>>> origin/main
