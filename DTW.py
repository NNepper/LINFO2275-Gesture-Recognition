import sklearn.exceptions

import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


def shrink_sequence(X):
    tmp = []
    for i in range(0, len(X) - (len(X) % 2), 2):
        tmp.append((X[i] + X[i + 1]) / 2)
    return np.array(tmp)


class DTW(BaseEstimator, ClassifierMixin):
    n_neighbors = np.Inf
    X = None
    labels = None
    window_size = np.Inf
    fitted = False
    shrinkage = True

    def __init__(self, n_neighbors=5, window_size=100, shrinkage=True):
        self.shrinkage = shrinkage
        self.n_neighbors = n_neighbors
        self.window_size = window_size

    def fit(self, X, y, shrinkage=True):
        self.fitted = True
        self.X = X
        self.labels = y

    # Dynamic Time Warping distance between sequence S1 and S2
    def _distance(self, s1, s2):
        if self.shrinkage:
            s1 = shrink_sequence(s1)
            s2 = shrink_sequence(s2)

        dtw = np.zeros(shape=(len(s1), len(s2)))
        w = max(self.window_size, abs(len(s1) - len(s2)))

        for i in range(0, len(s1)):
            for j in range(max(1, i - w), min(len(s2), i + w)):
                dtw[i, j] = np.Inf

        dtw[0, 0] = 0

        for i in range(1, len(s1)):
            for j in range(max(1, i - w), min(len(s2), i + w)):
                cost = np.linalg.norm(s1[i, :3] - s2[j, :3])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return dtw[-1, -1]

    def _distance_matrix(self, ms1, ms2):
        D = np.zeros((len(ms1), len(ms2)))

        max_count = len(ms1) * len(ms2)
        count = 0
        for i in range(len(ms1)):
            for j in range(len(ms2)):
                D[i, j] = self._distance(ms1[i], ms2[j])
                count += 1
                if count % 1000 == 0:
                    print("{}/{}".format(count, max_count))
        return D

    def predict(self, X):
        # Compute distance between train and test
        D = self._distance_matrix(self.X, X)
        sorted_idx = (-D).argsort(axis=0)[:self.n_neighbors]

        # label retrieval
        sorted_labels = np.array(
            [[self.labels[sorted_idx[j, i]] for i in range(len(X))] for j in range(self.n_neighbors)])
        vote = [np.bincount(sorted_labels[:, i]) for i in range(len(X))]
        return [np.argmax(vote[i]) for i in range(len(X))]

    def score(self, X, y, sample_weight=None):
        if self.fitted:
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
        else:
            raise sklearn.exceptions.NotFittedError


#####################
#       Launch      #
#####################
if __name__ == '__main__':
    # Data importing
    X, y = utils.read_files(1)
    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, 0.7)

    # Hyper-parameters tuning
    param_grid = [{
        "n_neighbors" : [3,4,5,7,10],
    }]
    model = DTW(window_size=10)
    search = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=4)
    search.fit(X_train, y_train)

    # Best model
    best_params = search.best_params_
    print(best_params)
    best_model = DTW(best_params)
    best_model.fit(X_train, y_train)
    print("accuracy on test datatset: {}".format(best_model.score(X_test, y_test)))
