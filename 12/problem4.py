import numpy as np
import matplotlib
from scipy import linalg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_data(sample_size, n_class):
    x = (np.random.normal(size=(sample_size//n_class, n_class)) + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class), (sample_size//n_class, n_class)).flatten()
    return x, y

def least_squares_prob_clr(x, y, h, l, n_class):
    sample_size = len(x)
    # theta = np.zeros((sample_size, n_class))

    Phi = np.exp(- (x[:, None] - x[None])**2 / (2 * h*h))
    Pi = np.zeros((sample_size, n_class))
    for i in range(sample_size):
        for c in range(n_class):
            if c == y[i]:
                Pi[i][c] = 1

    # print(y)
    # print("Pi", Pi)

    theta = np.linalg.solve(
        Phi.T.dot(Phi) + l * np.eye(sample_size),
        Phi.T.dot(Pi)
    )

    return theta


def visualize(x, y, theta, h):
    X = np.linspace(-5., 5., num=100)
    K = np.exp(-(x - X[:, None])**2 / (2 * h*h))
    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)
    prob = K.dot(theta)
    prob = np.where(prob > 0, prob, 0)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    # print(prob)
    # unnormalized_prob = np.exp(logit - np.max(logit, axis=1, keepdims=True))
    # prob = unnormalized_prob / unnormalized_prob.sum(1, keepdims=True)
    plt.plot(X, prob[:, 0], c='blue', label="p(y=0|x)")
    plt.plot(X, prob[:, 1], c='red', label="p(y=1|x)")
    plt.plot(X, prob[:, 2], c='green', label="p(y=2|x)")
    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')
    plt.legend()
    plt.savefig('12/fig.png')

x, y = generate_data(sample_size=90, n_class=3)
theta = least_squares_prob_clr(x, y, h=2., l=.1, n_class=3)
visualize(x, y, theta, h=2.)