#######################
### Utils functions ###
#######################

import numpy as np

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


# Resample the measurement to standardized batch
def train_test_split(X, y, train_ratio):
    cutting_pt = int(train_ratio * 10)
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(0, 1000, 10):
        sampling = np.arange(10)
        np.random.shuffle(sampling)

        for j in range(0, cutting_pt):
            X_train.append(X[i + sampling[j]])
            y_train.append(y[i + sampling[j]])
        for j in range(cutting_pt, 10):
            X_test.append(X[i + sampling[j]])
            y_test.append(y[i + sampling[j]])
    return np.array(X_train, dtype=object), np.array(y_train, dtype=object), np.array(X_test, dtype=object), np.array(
        y_test, dtype=object)
