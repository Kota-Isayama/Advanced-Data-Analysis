import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 0
    y[n_positive:] = 1
    return x, y

def cwls(train_x, train_y, test_x):
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    print(np.linalg.norm(train_x[train_y==0][None] - train_x[train_y==1][:,None,:], axis=2).shape)
    for i in range(2):
        for j in range(2):
            A[i][j] = np.linalg.norm(train_x[train_y==i][None] - train_x[train_y==j][:,None,:], axis=2).mean()
    for i in range(2):
        b[i] = np.linalg.norm(test_x[None] - train_x[train_y==i][:,None,:], axis=2).mean()
    pi = (A[0][1] - A[1][1] - b[0] + b[1]) / (2*A[0][1] - A[0][0] - A[1][1])
    # print(pi)
    # pi = np.min(1.0, np.amax(0.0, pi))

    num_0 = (train_y==0).sum()
    num = train_y.shape[0]
    r = np.where(train_y==0, pi/num_0, (1-pi)/(num-num_0))
    print(r)
    r = np.sqrt(r)
    t_x = train_x.copy()
    t_y = train_y.copy()
    t_x = np.hstack((t_x, np.ones_like(t_y).reshape(-1, 1)))
    t_x = r.reshape(-1, 1) * t_x
    t_y = np.where(t_y == 0, +1, -1)
    t_y = r * t_y

    theta = np.linalg.solve(
        t_x.T.dot(t_x),
        t_x.T.dot(t_y)
    )
    return theta

def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6,6))
        plt.xlim(-5, 5)
        plt.ylim(-7, 7)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y==0][:, 0], x[y==0][:, 1], marker='$O$', c='blue')
        plt.scatter(x[y==1][:, 0], x[y==1][:, 1], marker='$X$', c='red')
        plt.savefig('./lecture8-h3-{}.png'.format(name))

train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)