import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


class Dollar(BaseEstimator, ClassifierMixin):
    X = None
    labels = None

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.labels = y

    def predict(self, X):
        return np.zeros(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)


if __name__ == '__main__':
    X, y = utils.read_files(domain=1)

    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, 0.7)

    model = Dollar()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
