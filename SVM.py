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


def plot_facet(X1, X2):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    index1 = np.array(X1.index)
    index2 = np.array(X2.index)

    ax1.set_title('Sequence in [x,y,z]')

    ax1.plot(index1, X1["x"], "tab:red")
    ax1.plot(index2, X2["x"], "tab:orange")
    ax2.plot(index1, X1["y"], 'tab:red"')
    ax2.plot(index2, X2["y"], "tab:orange")
    ax3.plot(index1, X1["z"], "tab:red")
    ax3.plot(index2, X2["z"], "tab:orange")

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.show()


def dataframe_conversion(X):
    X_new = []
    for i in range(len(X)):
        X[i] = X[i][3:-3]
        df = pd.DataFrame(X[i], columns=["x", "y", "z", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        X_new.append(df)
    return X_new


class SVM(BaseEstimator, ClassifierMixin):
    X = None
    labels = None
    resampling_size = None
    conv_threshold = None

    def __init__(self, resampling_size=64, threshold=0.06):
        self.resampling_size = resampling_size
        self.conv_threshold = threshold

    def fit(self, X, y):
        self.X = self._feature_creation(self._preprocessing(dataframe_conversion(X)))
        self.labels = y

    def predict(self, X):
        return np.zeros(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)

    def _preprocessing(self, X):

        # Resample in constant step (N=64)
        for i in range(len(X)):
            X[i] = resample(X[i], self.resampling_size)

        # Remove outlier with threshold
        for i in range(len(X)):
            # Determine rejection threshold
            threshold = 2 * rejection_threshold(X[i][["x", "y", "z"]])
            to_drop = []

            # Outlier rejection
            for col in ["x", "y", "z"]:
                tmp_conv = gaussian_filter1d(X[i][col], 5)
                tmp_base = list(X[i][col])
                for j in range(len(X[i])):
                    if np.abs(tmp_base[j] - tmp_conv[j]) > threshold:
                        to_drop.append(j)
            if len(to_drop) > 0:
                X[i] = X[i].drop(labels=[X[i].index[idx] for idx in set(to_drop)], axis=0)
                X[i] = resample(X[i], self.resampling_size)

        return X

    def _feature_creation(self, X):
        # TODO: feature creation for SVM classifier

        return X  # Return matrix of new features


if __name__ == '__main__':
    X, y = utils.read_files(domain=1)

    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, 0.7)

    model = SVM()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
