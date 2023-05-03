from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

def calc_design_matrix(x, c, h):
    return np.exp(-((x[:, None] - c[None]) ** 2 / (2 * (h ** 2))))

def iterative_reweighted_shrinkage(x, y, h, l=.1, n_iter=1000):
    phi = calc_design_matrix(x, x, h)
    theta = theta_prev = np.linalg.solve(
        phi.T.dot(phi) + 1e-4 * np.identity(phi.shape[1]),
        phi.T.dot(y)
    )
    for _ in range(n_iter):
        # Theta_inv = np.diag(np.where(np.abs(theta_prev) > 1e-3, 1.0 / np.abs(theta_prev), 0))
        Theta_inv = np.linalg.pinv(np.diag(np.abs(theta_prev)))
        theta = np.linalg.solve(
            phi.T.dot(phi) + l * Theta_inv,
            phi.T.dot(y)
        )
        print("{}th iterate : error {:.6f}: obj {:.6f}".format(
            _,
            np.linalg.norm(theta - theta_prev),
            np.linalg.norm(phi.dot(theta) - y) / 2 + l * np.abs(theta).sum() / 2
        ))
        if np.linalg.norm(theta - theta_prev) < 1e-3:
            break
        theta_prev = theta
    return theta

# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# caluculate design matrix
h = 0.1
k = calc_design_matrix(x, x, h)
l = 5

theta = iterative_reweighted_shrinkage(x, y, h, l)

print("the number of zeros: {}".format(
    (theta < 1e-3).astype('int32').sum()
))

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(X, x, h)
prediction = K.dot(theta)

# visualization
plt.clf()
plt.scatter(x, y, c='green', marker='o', label='training samples')
plt.plot(X, prediction, label='predictions')
plt.xlabel('x (expolanatory variable)')
plt.ylabel('y (objective variable)')
plt.legend()
plt.savefig('homework3-1_' + str(l).replace('.', '') + '.png')