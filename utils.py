import numpy as np
import scipy.interpolate as interp


#######################
### Utils functions ###
#######################

# Read specific filename from specified domain
def read_files(domain):
    X = []
    y = []
    for i in range(1, 1001):
        path = "data/Domain0{}/{}.txt".format(domain, i)

        with open(path, "r") as file:
            lines = file.readlines()

            # Store class
            clas = int(lines[1].split()[-1].rstrip("\n")) - 1

            y.append(clas)

            # Store Sequence
            tmp = []
            for i in range(5, len(lines)):
                tmp.append([float(val) for val in lines[i].rstrip("\n").split(",")])
            X.append(np.matrix(tmp))
    return X, y


# Split Dataset in Training and Testing set
def train_test_split(X, y):
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(X) - 100):
        X_train.append(X[i])
        y_train.append(y[i])
    for i in range(len(X) - 100, len(X)):
        X_test.append(X[i])
        y_test.append(y[i])
    return np.array(X_train, dtype=object), np.array(y_train, dtype=object), \
           np.array(X_test, dtype=object), np.array(y_test, dtype=object)


# Resampling function
def resample(X, n_new=64):
    n_old, m = X.shape
    mat_new = np.zeros((n_new, m))
    x_old = np.asarray(X[:, 3]).squeeze()
    x_new = np.linspace(X[:, 3].min(), X[:, 3].max(), n_new)

    for j in range(m - 1):
        y_old = np.asarray(X[:, j]).squeeze()
        interpolator = interp.interp1d(x_old, y_old)
        y_new = interpolator(x_new)
        mat_new[:, j] = y_new
    mat_new[:, -1] = x_new
    return mat_new


def score(y_pred, y_true):
    count = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            count += 1
    return count / len(y_pred)


# Leave-One-Out Cross-Validation
class LeaveOneOut:

    def __init__(self):
        return

    def split(self, X, y, groups=None):
        for i in range(100, 1000, 100):  # len(x) folds
            right_train = np.arange(i, len(X))
            if i > 100:
                left_train = np.arange(0, i - 100)
            else:
                left_train = np.array([])
            train_idx = np.concatenate([left_train, right_train])
            val_idx = np.arange(i - 100, i)
            yield np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)

    def get_n_splits(self, X, y, groups=None):
        return 10
