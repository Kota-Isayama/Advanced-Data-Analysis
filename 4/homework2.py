import csv
import numpy as np

def load_data_from_csv(filename):
    data = list()
    y = list()
    for i in range(10):
        with open(filename.format(i), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data_i = np.array([[float(v) for v in line] for line in reader])
            y.append(i for _ in range(len(data_i)))

    return data, y

def build_design_mat(x1, x2, bandwidth):
    return np.exp(-np.sum(x1[:, None] - x2[None])**2, axis=2) / (2 * bandwidth**2)

def optimize_param(design_mat, y, regularizer):
    thetas = list()
    hat_matrix = design_mat.T.dot(design_mat) + regularizer * np.identity(len(y))
    for c in range(10):
        thetas.append(np.linalg.solve(
            hat_matrix,
            design_mat.T.dot(np.where(y == c, 1, -1))
        ))
    return np.stack(thetas, axis=1)

def predict(train_data, test_data, thetas):
    return np.argmax(build_design_mat(train_data, test_data, 10.).T.dot(thetas), axis=1)

def build_confusion_matrix(train_data, test_data, test_y, thetas):
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    correct = 0
    for i in range(10):
        data = test_data[test_y == i]
        prediction = predict(train_data, data, thetas)
        for j in range(10):
            confusion_matrix[i][j] = np.sum(np.where(prediction == j, 1, 0))
        correct += confusion_matrix[i][i]
    accuracy = correct / len(test_data)
    return accuracy, confusion_matrix

# multi-class classification based on one-versus-all
train_data, train_label = load_data_from_csv('digit_train{}.csv')
x, y = np.concatenate(train_data), np.concatenate(train_label)
x, y = x[::4], y[::4] # subsample the data for faster computation
design_mat = build_design_mat(x, x, 10.)
thetas = optimize_param(design_mat, y, 1.)
test_data, test_label = load_data_from_csv('digit_test{}.csv')
test_x, test_y = np.concatenate(test_data), np.concatenate(test_label)
accuracy, confusion_matrix = build_confusion_matrix(x, test_x, test_y, thetas)
print('accuracy: {}'.format(accuracy))
print('confusion matrix:')
print(confusion_matrix)