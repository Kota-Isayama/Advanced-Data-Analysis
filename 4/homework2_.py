import numpy as np
from scipy.io import loadmat

def get_raw_data(file_name):
    data = loadmat(file_name)
    train = data['X']
    test = data['T']
    return data, train, test

def process_raw_data(x):
    x_ = np.copy(x).transpose((2, 1, 0))
    return x_

def calc_design_mat(x, c, h):
    return np.exp(-np.sum((x[:, None] - c[None])**2, axis=2) / (2*h**2))

def train_model(x, y, lam):
    tmp_x = np.copy(x).reshape(x.shape[0]*x.shape[1], -1)
    tmp_y = np.copy(y).flatten()
    Phi = calc_design_mat(tmp_x, tmp_x, 1.)
    thetas = list()
    print(Phi.shape)
    print(len(y))
    A = Phi.T.dot(Phi) + lam * np.identity(len(tmp_y))
    for i in range(10):
        thetas.append(np.linalg.solve(
            A, Phi.T.dot(np.where(tmp_y == i, 1, -1))
        ))
    return np.stack(thetas, axis=0)

def predict(train_x, test_x, thetas):
    tmp_train_x = np.copy(train_x).reshape((train_x.shape[0] * train_x.shape[1], -1))
    tmp_test_x = np.copy(test_x).reshape((test_x.shape[0] * test_x.shape[1], -1))
    Phi = calc_design_mat(tmp_test_x, tmp_train_x, 10.)
    return Phi.dot(thetas.T)

def calc_confusion_mat(train_data, test_data, test_y, thetas):
    confusion_mat = np.zeros((10, 10))
    correct = 0
    for i in range(10):
        data = test_data[test_y == i]
        pred = predict(train_data, data, thetas)
        for j in range(10):
            confusion_mat[i][j] = np.sum(np.where(pred == j, 1, 0))
        correct += confusion_mat[i][i]
    acc = correct / len(test_data)
    return acc, confusion_mat

data, train, test = get_raw_data("4/ADA4-digit.mat")
train_x = process_raw_data(train)
test_x = process_raw_data(test)
train_y = np.zeros((train_x.shape[0], train_x.shape[1]))
print(train_y.shape)
for i in range(10):
    train_y[i] = i
test_y = np.zeros((test_x.shape[0], test_x.shape[1]))
for i in range(10):
    test_y[i] = i

train_x = train_x[:, ::10]
train_y = train_y[:, ::10]

Phi = calc_design_mat(train_x, train_x, 10.)
thetas = train_model(train_x, train_y, 1.)

acc, confusion_mat = calc_confusion_mat(train_x, test_x, test_y, thetas)
print('accuracy: {}'.format(acc))
print('confusion matrix:')
print(confusion_mat)