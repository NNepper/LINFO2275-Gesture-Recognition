import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resample(X, n_new=64):
    X_res = []
    for df in X:
        n_old, m = df.values.shape
        mat_old = df.values
        mat_new = np.zeros((n_new, m))
        x_old = np.linspace(df.index.min(), df.index.max(), n_old)
        x_new = np.linspace(df.index.min(), df.index.max(), n_new)

        for j in range(m-1):
            y_old = np.array(mat_old[:, j], dtype=float)
            y_new = np.interp(x_new, x_old, y_old)
            mat_new[:, j] = y_new

        X_res.append(pd.DataFrame(mat_new, index=x_new, columns=df.columns))
    return X_res



class SVM(BaseEstimator, ClassifierMixin):
    X = None
    labels = None
    resampling_size = None
    conv_threshold = None

    def __init__(self, resampling_size=64, threshold=0.06):
        self.resampling_size = resampling_size
        self.conv_treshold = threshold

    def fit(self, X, y):
        X_pre = self._preprocessing(X)
        self.X = self._feature_creation(X_pre)
        self.labels = y

    def predict(self, X):
        return np.zeros(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)

    def _preprocessing(self, X):
        X_pre = []
        # Remove first and last hooking
        for i in range(len(X)):
            X[i] = X[i][3:-3]
            df = pd.DataFrame(X[i], columns=["x", "y", "z", "timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            X_pre.append(df)

        # Resample in constant step (N=64)
        for i in range(len(X)):
            X_pre[i] = resample(X_pre, self.resampling_size)

        # Remove outlier with threshold
        for i in range(len(X_pre)):
            for col in ["x", "y", "z"]:
                tmp = gaussian_filter1d(X[i][col], 5)
                for j in range(len(X[i])):
                    if np.abs(X_pre[i][col][j] - tmp[j]) > self.conv_treshold:
                        X_pre[i].drop(index=j)
            if len(X_pre[i]) != self.resampling_size:
                X_pre[i] = resample(X_pre[i], self.resampling_size)

        return X_pre

    def _feature_creation(self, X):
        # TODO: feature creation for SVM classifier

        return X  # Return matrix of new features


if __name__ == '__main__':
    X, y = utils.read_files(domain=1)

    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, 0.7)

    model = SVM()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
