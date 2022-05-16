import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


###############################
### Preprocessing functions ###
###############################

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





##################################
### Feature creation functions ###
##################################

def speed_feature(X):
    X_speed = []
    for seq in X:
        length = [0.0]
        for j in range(1, len(seq)):
            vec1 = seq[j, :][0:2]
            vec2 = seq[j - 1, :]
            length[j] = np.linalg.norm(vec2 - vec1) + length[j - 1]

        speed = []
        for j in range(1, len(seq) - 1):
            speed.append(
                (length[j + 1] - length[j - 1]) / (seq.loc[j + 1] - seq.loc[j - 1]))  # Compute speed at point j
        speed_smoothed = gaussian_filter1d(speed, 1)
        features = np.array([
            np.std(speed_smoothed)/np.mean(speed_smoothed),
            np.percentile(speed_smoothed, q=0.9)/np.mean(speed_smoothed),
            np.percentile(speed_smoothed, q=0.1)/np.mean(speed_smoothed)
        ])
        X_speed.append(features)
    return X_speed

def curvature_feature(X):
    X_curv = []
    for seq in X:
        curv = []
        for i in range(len(seq)):
            # Compute sequence length
            for j in range(1, len(X[i])):
                vec1 = X[i][j - 1, :][0:2]
                vec2 = X[i][j, :][0:2]
                curv.append(np.linalg.norm(vec1, vec2) / math.acos(np.dot(vec1, vec2)))
        curv_smoothed = gaussian_filter1d(curv, 1)
        features = np.array([
            np.mean(curv_smoothed),
            np.std(curv_smoothed),
            np.quantile(curv_smoothed, q=0.9)
        ])
        X_curv.append(features)
    return X_curv

def distance_feature(X):
    X_dist = []
    for seq in X:
        dist = np.zeros(shape=(len(seq), len(seq)))
        for i in range(len(seq)):
            for j in range(len(seq)):




class SVM(BaseEstimator, ClassifierMixin):
    X = None
    labels = None
    resampling_size = None
    conv_threshold = None

    def __init__(self, resampling_size=72, threshold=0.16):
        self.resampling_size = resampling_size
        self.conv_threshold = threshold

    def fit(self, X, y):
        self.X = self._feature_creation(self._preprocessing(X))
        self.labels = y

    def predict(self, X):
        return np.zeros(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)

    def _preprocessing(self, X):

        # Resample in constant step (N=64)
        for i in range(len(X)):
            X[i] = utils.resampling(X[i], self.resampling_size)

        # Remove outlier with threshold
        for i in range(len(X)):
            # Determine rejection threshold
            threshold = self.conv_threshold * rejection_threshold(X[i][0:2])
            to_drop = []

            # Outlier rejection
            for col in range(3):
                tmp_conv = gaussian_filter1d(X[i][:, col], sigma=5)
                tmp_base = list(X[i][:, col])
                for j in range(len(X[i])):
                    if distance.euclidean(tmp_base, tmp_conv) > threshold:
                        to_drop.append(j)
            if len(to_drop) > 0:
                X[i] = np.delete(X[i], list(set(to_drop)), axis=0)
                X[i] = utils.resampling(X[i], self.resampling_size)

        # Scale path-length to unity
        for i in range(len(X)):
            # Compute sequence length
            length = 0.0
            for j in range(1, len(X[i])):
                vec1 = X[i][j - 1, :][0:2]
                vec2 = X[i][j, :][0:2]
                length += np.linalg.norm(vec2 - vec1)
            X[i][0:2] /= length

        print("Preprocessing Done!")
        return X

    def _feature_creation(self, X):
        # TODO: feature creation for SVM classifier

        # TODO: Speed feature (f1,f2,f3)
        for seq in range(len(X)):
            length = 0.0

        # TODO:
        return X  # Return matrix of new features


if __name__ == '__main__':
    # Data importing
    X, y = utils.read_files(1)
    X_train, y_train, X_test, y_test = utils.train_test_split(X, y)

    # Hyper-parameters tuning
    model = SVM()
    cv = utils.LeaveOneOut()
    score = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        model.fit(np.take(X_train, train_idx), np.take(y_train, train_idx))
        score.append(model.score(np.take(X_train, val_idx), np.take(y_train, val_idx)))
    print("mean accuracy score = %f" % np.array(score).mean())
