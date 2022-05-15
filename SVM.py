import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resample(df, n_new=64):
    n_old, m = df.values.shape
    mat_old = df.values
    mat_new = np.zeros((n_new, m))
    x_old = np.linspace(df.index.min(), df.index.max(), n_old)
    x_new = np.linspace(df.index.min(), df.index.max(), n_new)

    for j in range(m - 1):
        y_old = np.array(mat_old[:, j], dtype=float)
        y_new = np.interp(x_new, x_old, y_old)
        mat_new[:, j] = y_new

    return pd.DataFrame(mat_new, index=x_new, columns=df.columns)


def rejection_threshold(X):
    X_pca = PCA().fit_transform(X)
    return np.linalg.norm(X_pca[0].max() - X_pca[0].min())


def plot_facet(X):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(X.index, X["x"])
    ax1.set_title('Sequence in [x,y,z]')
    ax2.plot(X.index, X["y"])
    ax3.plot(X.index, X["z"])
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.show()


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
            X_pre[i] = resample(X_pre[i], self.resampling_size)

        # Remove outlier with threshold
        for i in range(len(X_pre)):
            # Determine rejection threshold
            threshold = 2 * rejection_threshold(X_pre[i][["x", "y", "z"]])
            to_drop = []

            # Outlier rejection
            for col in ["x", "y", "z"]:
                tmp_conv = gaussian_filter1d(X_pre[i][col], 5)
                tmp_base = list(X_pre[i][col])
                for j in range(len(X_pre[i])):
                    if np.abs(tmp_base[j] - tmp_conv[j]) > threshold:
                        to_drop.append(j)
            if len(to_drop) > 0:
                X_pre[i] = X_pre[i].drop(labels=[X_pre[i].index[idx] for idx in set(to_drop)], axis=0)
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
