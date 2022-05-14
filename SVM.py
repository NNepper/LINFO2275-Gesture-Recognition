import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


class SVM(BaseEstimator, ClassifierMixin):
    X = None
    labels = None


    def __init__(self):
        pass

    def fit(self, X, y):
        X_pre= self._preprocessing(X)
        self.X = self._feature_creation(X_pre)
        self.labels = y

    def predict(self, X):
        return np.zeros(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)

    def _preprocessing(self, X):
        #TODO: remove first and last points

        #TODO: resample in constant step (N=64)

        #TODO: Compare X to X_conv with threshold to remove outlier

        #TODO: Resample again (N=64)

        return X

    def _feature_creation(self, X):
        #TODO: feature creation for SVM classifier

        return X #Return matrix of new features

if __name__ == '__main__':
    X, y = utils.read_files(domain=1)

    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, 0.7)

    model = SVM()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
