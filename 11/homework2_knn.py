import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
np.random.seed(1)

def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)], axis=1)

def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)

def data_generation3(n=100):
    x = np.concatenate([np.linspace(-2, 2, n).reshape(n, -1), np.ones((n, 1)) * (1 - 2 * (np.arange(n) % 2)).reshape(n, -1)], axis=1)
    # return np.where(np.arange(x.shape[0]) % 2, -x, x)
    print(x)
    return x

def similarity_matrix_knn(x, k=5):
    d = np.linalg.norm(x[None] - x[:, None, :], ord=2, axis=2)
    indices = np.argsort(d, axis=1)[:,:k]
    W = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        W[i, indices[i]] = 1.0
        W[indices[i], i] = 1.0
    return W


def lpp(x, n_components=1):
    x -= np.mean(x, axis=0)
    # W = similarity_matrix(x)
    W = similarity_matrix_knn(x, 30)
    print(W.sum(axis=1))
    d = np.sum(W, axis=1)
    D = np.diag(d)
    L = D - W
    A = x.T.dot(L).dot(x)
    B = x.T.dot(D).dot(x)
    # rB = sp.linalg.sqrtm(B)
    # irB = np.linalg.inv(rB)
    # C = irB.dot(A).dot(irB)
    # w, v = sp.linalg.eigh(C)
    w, v = sp.linalg.eigh(A, B)
    print(w)
    print("labmda")
    print(A.dot(v) / B.dot(v))
    print(v)
    print(np.linalg.norm(v, 2, 0, keepdims=True))
    v = v / np.linalg.norm(v, 2, 0, keepdims=True)
    print("v")
    print(v)
    print(np.linalg.norm(v, 2, 0))
    return w[:n_components], v[:,:n_components]

n = 100
n_components = 1
# x = data_generation3(n)
x = data_generation2(n)
# print(x.shape)

w, v = lpp(x, n_components)

# lpp = LPP(n_components)
# v = lpp.fit(x)

plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[:, 0], x[:, 1], 'rx')
# plt.show()
# plt.plot(np.array([-v[:,0], v[:,0]]) * 9, np.array([-v[:,1], v[:,1]]) * 9)
plt.plot(np.array([-v[0], v[0]]) * 9, np.array([-v[1], v[1]]) * 9)
plt.savefig("11/lpp_homework_knn_30.png")
