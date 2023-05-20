import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.random.seed(1)

def generate_data(sample_size):
    """Generate training data.
    
    Since
    f(x) = w^{T}x + b
    can be written as 
    f(x) = (w^{T}, b)(x^{T}, 1)^{T},
    for the sake of simpler implementation of SVM,
    we return (x^{T}, 1)^{T} instead of x

    :param sample_size: number of data points in the sample
    :return: a tuple of data point and label
    """

    x = np.random.normal(size=(sample_size, 3))
    x[:, 2] = 1.
    x[:sample_size//2, 0] -= 5
    x[sample_size//2:, 0] += 5
    y = np.concatenate([np.ones(sample_size//2, dtype=np.int64),
                        -np.ones(sample_size//2, dtype=np.int64)])
    
    x[:3, 1] -= 5.
    y[:3] = -1
    x[-3:, 1] += 5
    y[-3:] = 1
    return x, y

def svm(x, y, l, lr):
    """Linear SMV implementation using gradient descent algorithm
    
    f_w(x) = w^{T} (x^{T}, 1)^{T}

    :param x: data points
    :param y: label
    :param l: regularization parameter
    :param lr: learning rate
    :return: three-dimensional vector w
    """

    w = np.zeros(3)
    prev_w = w.copy()
    for i in range(10 ** 4):

        f = x.dot(w)
        fy = f * y
        fy = np.stack([fy, fy, fy]).T
        # print("fy.shape: {}, {}".format(fy.shape[0], fy.shape[1]))
        xy = x * y[:, None]
        # print("xy.shape: {}, {}".format(xy.shape[0], xy.shape[1]))
        dw = l * np.where(np.abs(fy-1) < 1e-3, -xy/2, np.where(fy > 1, 0, -xy)).sum(axis=0)
        dw[:2] = dw[:2] + 2 * w[0:2]
        
        w = prev_w - lr * dw

        if np.linalg.norm(w - prev_w) < 1e-3:
            break
        prev_w = w.copy()

    return w

def visualize(x, y, w):
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.scatter(x[y==1, 0], x[y==1, 1])
    plt.scatter(x[y==-1, 0], x[y==-1, 1])
    plt.plot([-10, 10], -(w[2] + np.array([-10, 10]) * w[0]) / w[1])
    plt.savefig('lecture6_h2.png')

x, y = generate_data(200)
w = svm(x, y, l=5., lr=.1)
visualize(x, y, w)