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
            s1 = utils.resampling(s1, 64)
            s2 = utils.resampling(s2, 64)

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

        for i in range(len(ms1)):
            for j in range(len(ms2)):
                D[i, j] = self._distance(ms1[i], ms2[j])
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
            count = 0
            for i in range(len(y_pred)):
                if y[i] == y_pred[i]:
                    count += 1
            return count / len(y_pred)
        else:
            raise sklearn.exceptions.NotFittedError

    def set_params(self, **params):
        for param, val in params.items():
            setattr(self, param, val)


#####################
#       Launch      #
#####################
if __name__ == '__main__':
    # Data importing
    X, y = utils.read_files(1)
    X_train, y_train, X_test, y_test = utils.train_test_split(X, y)

    # Hyper-parameters tuning
    param_grid = [{
        "n_neighbors": [3, 4, 5, 7, 10],
    }]
    model = DTW(window_size=10, shrinkage=True)
    cv = utils.LeaveOneOut()
    for n_neighbor in [3, 4, 5, 7, 10]:
        score = []
        count = 0
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(np.take(X_train, train_idx), np.take(y_train, train_idx))
            score.append(model.score(np.take(X_train, val_idx), np.take(y_train, val_idx)))
            print("iteration n°{} for n_neighbors={} gave score:{}".format(count, n_neighbor, score[-1]))
            count += 1
        print("mean accuracy score for n:{} = {}".format(n_neighbor, np.array(score).mean()))
