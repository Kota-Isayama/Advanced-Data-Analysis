from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)

def generate_sample(xmin=-3, xmax=3, sample_size=10):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    y = x + np.random.normal(loc=0., scale=.2, size=sample_size)
    y[-1] = y[-2] = y[1] = -4
    return x, y

def calc_design_matrix(x):
    sample_size = len(x)
    phi = np.empty(shape=(sample_size, 2))
    phi[:, 0] = 1
    phi[:, 1] = x
    return phi

def iterative_reweighted_least_squares(x, y, eta=1., n_iter=1000):
    phi = calc_design_matrix(x)
    theta = theta_prev = np.linalg.solve(
        phi.T.dot(phi) + 1e-4 * np.identity(phi.shape[1]),
        phi.T.dot(y)
    )
    for _ in range(n_iter):
        r = np.abs(phi.dot(theta_prev) - y)
        w = np.diag(np.where(r > eta, 0, (1 - (r ** 2) / (eta ** 2)) ** 2))
        phit_w_phi = phi.T.dot(w).dot(phi)
        phit_w_y = phi.T.dot(w).dot(y)
        theta = np.linalg.solve(phit_w_phi, phit_w_y)
        print("{}th iterate : error {:.6f}".format(
            _,
            np.linalg.norm(theta - theta_prev)
        ))
        if np.linalg.norm(theta - theta_prev) < 1e-3:
            break
        theta_prev = theta
        
    return theta

sample_size = 10
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

theta = iterative_reweighted_least_squares(x, y, eta=1)

X = np.linspace(start=xmin, stop=xmax, num=5000)
Phi = calc_design_matrix(X)
prediction = Phi.dot(theta)

plt.clf()
plt.scatter(x, y, c='green', marker='o', label='training samples')
plt.plot(X, prediction, label='learned function')
plt.legend()
plt.savefig('homework3-3_eta_1.png')