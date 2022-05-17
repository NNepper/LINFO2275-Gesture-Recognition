import copy

import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.ndimage import gaussian_filter1d
import scipy.spatial.distance as distance
import math
import copy
import numpy as np


###############################
### Preprocessing functions ###
###############################

def rejection_threshold(X):
    X_pca = PCA().fit_transform(X)
    x_min = [X_pca[0].min(), X_pca[1].min(), X_pca[2].min()]
    x_max = [X_pca[0].max(), X_pca[1].max(), X_pca[2].max()]
    return math.dist(x_min, x_max)


##################################
### Feature creation functions ###
##################################

def acceleration_feature(X):
    X_acceleration = []
    for seq in X:
        # First derivative
        dx_dt = np.gradient(seq[:, 0])
        dy_dt = np.gradient(seq[:, 1])
        dz_dt = np.gradient(seq[:, 2])
        ds_dt = np.sqrt((dx_dt - dy_dt) ** 2 + (dx_dt - dz_dt) ** 2 + (dy_dt - dx_dt) ** 2)
        velocity = np.array([[dx_dt[i], dy_dt[i], dz_dt[i]] for i in range(dx_dt.size)])

        # Second derivative
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)
        d2s_dt2 = np.gradient(ds_dt)

        # Unit tangent vector
        tangent = np.array([1 / ds_dt] * 3).transpose() * velocity
        dtangent_x = np.gradient(tangent[:, 0])
        dtangent_y = np.gradient(tangent[:, 1])
        dtangent_z = np.gradient(tangent[:, 2])
        dT_dt = np.array([[dtangent_x[i], dtangent_y[i], dtangent_z[i]] for i in range(dtangent_x.size)])
        norm_tangent = np.sqrt(dtangent_x ** 2 + dtangent_y ** 2 + dtangent_z ** 2)

        # Unit normal vector
        normal = np.zeros(shape=(len(seq), 3))
        for i in range(len(norm_tangent)):
            if norm_tangent[i] != 0.0:
                normal[i, :] = dT_dt[i] / norm_tangent[i]

        # Curvature
        curvature = (d2z_dt2 - dy_dt) ** 2 + (d2x_dt2 - d2z_dt2) ** 2 + (d2y_dt2 - d2x_dt2) ** 2
        curvature = np.sqrt(curvature) / (dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2) ** 1.5

        # Acceleration
        t_component = np.array([d2s_dt2] * 3).transpose()
        n_component = np.array([curvature * ds_dt * ds_dt] * 3).transpose()
        acceleration = t_component * tangent + n_component * normal

        # Smoothing
        acceleration_smoothed = gaussian_filter1d(acceleration, 1)
        features = np.array([
            np.std(acceleration_smoothed) / np.mean(acceleration_smoothed),
            np.percentile(acceleration_smoothed, q=0.9) / np.mean(acceleration_smoothed),
            np.percentile(acceleration_smoothed, q=0.1) / np.mean(acceleration_smoothed)
        ])
        X_acceleration.append(features)
    return np.array(X_acceleration)


def speed_feature(X):
    X_speed = []
    for i in range(len(X)):
        seq = X[i]
        dx_dt = np.gradient(np.asarray(seq[:, 0]).squeeze())
        dy_dt = np.gradient(np.asarray(seq[:, 1]).squeeze())
        dz_dt = np.gradient(np.asarray(seq[:, 2]).squeeze())

        ds_dt = np.sqrt((dx_dt - dy_dt) ** 2 + (dx_dt - dz_dt) ** 2 + (dy_dt - dx_dt) ** 2)

        speed_smoothed = gaussian_filter1d(ds_dt, 1)
        features = np.array([
            np.std(speed_smoothed) / np.mean(speed_smoothed),
            np.percentile(speed_smoothed, q=0.9) / np.mean(speed_smoothed),
            np.percentile(speed_smoothed, q=0.1) / np.mean(speed_smoothed)
        ])
        X_speed.append(features)
    return np.array(X_speed)


def curvature_feature(X):
    X_curv = []
    for seq in X:
        # 1st derivative
        dx_dt = np.gradient(np.asarray(seq[:, 0])).squeeze()
        dy_dt = np.gradient(np.asarray(seq[:, 1])).squeeze()
        dz_dt = np.gradient(np.squeeze(seq[:, 2])).squeeze()

        # 2nd derivative
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)

        # Curvature
        kk = (d2z_dt2 - dy_dt) ** 2 + (d2x_dt2 - d2z_dt2) ** 2 + (d2y_dt2 - d2x_dt2) ** 2
        kk = np.sqrt(kk) / (dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2) ** 1.5

        # Smoothing
        curv_smoothed = gaussian_filter1d(kk, 1)
        features = np.array([
            np.mean(curv_smoothed),
            np.std(curv_smoothed),
            np.quantile(curv_smoothed, q=0.9),
        ])
        X_curv.append(features)
    return np.array(X_curv)


def distance_feature(X):
    X_dist = []
    K = 16
    M = 4
    for seq in X:
        dist = np.zeros(shape=(len(seq), len(seq)))
        for i in range(len(seq)):
            for j in range(len(seq)):
                dist[i, j] = np.linalg.norm(seq[i, 0:3] - seq[j, 0:3])
        dist_averaged = np.zeros(shape=(K, K))
        for u in range(dist_averaged.shape[0]):
            for v in range(dist_averaged.shape[1]):
                cum_sum = 0.0
                for i in range((u - 1) * (M + 1), u * M):
                    for j in range((v - 1) * (M + 1), v * M):
                        cum_sum += dist[i, j]
                dist_averaged[u, v] = cum_sum / M ** 2
        X_dist.append(np.ravel(dist_averaged))
    return np.array(X_dist)


def angle_feature(X):
    X_angle = []
    K = 8
    M = 8
    for i in range(len(X)):
        seq = X[i]
        angle = np.zeros(shape=(len(seq), len(seq)))
        q = np.zeros(shape=(len(seq), 3))
        for i in range(1, len(seq)):
            q[i] = seq[i, 0:3] - seq[i - 1, 0:3]
        for i in range(len(seq) - 1):
            for j in range(len(seq) - 1):
                norm = np.linalg.norm(q[i, 0:3]) * np.linalg.norm(q[j, 0:3])
                if norm != 0.0:
                    tmp = min((q[i, 0:3] @ q[j, 0:3]) / norm, 1)
                    angle[i, j] = np.rad2deg(np.arccos(tmp))

        angle[:, -1] = angle[:, -2]
        angle[-1, :] = angle[-2, :]
        angle_averaged = np.zeros(shape=(K, K))
        for u in range(angle_averaged.shape[0]):
            for v in range(angle_averaged.shape[1]):
                cum_sum = 0.0
                for i in range((u - 1) * (M + 1), u * M):
                    for j in range((v - 1) * (M + 1), v * M):
                        cum_sum += angle[i, j]
                angle_averaged[u, v] = cum_sum / M ** 2
        X_angle.append(np.ravel(angle_averaged))
    return np.array(X_angle)


class SVM(BaseEstimator, ClassifierMixin):
    X = None
    labels = None
    resampling_size = None
    conv_threshold = None
    model = None
    pca_feature = None

    def __init__(self, C, gamma, kernel, resampling_size=64, threshold=0.06):
        # Preprocessing Hyper-parameters
        self.resampling_size = resampling_size
        self.conv_threshold = threshold
        # Support Vector Machine
        self.model = SVC(C=C, gamma=gamma, kernel=kernel)

    def fit(self, X, y):
        X_preprocess = self._preprocessing(X)
        self.X = self._feature_creation(X_preprocess)
        self.labels = y.astype('int')
        self.model.fit(self.X, self.labels)

    def predict(self, X):
        X_pred = self._feature_creation(self._preprocessing(X))
        return self.model.predict(X_pred)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)

    def _preprocessing(self, X):
        X_pre = copy.deepcopy(X)  # Preprocessed X

        # Remove outlier with threshold
        for i in range(len(X_pre)):
            # Determine rejection threshold
            X_pre[i] = X_pre[i][5:-5, :]
            threshold = rejection_threshold(X_pre[i][:, 0:3]) * self.conv_threshold
            to_drop = []

            # Outlier rejection
            for col in range(3):
                tmp_conv = gaussian_filter1d(X_pre[i][:, col], sigma=5)
                tmp_base = list(X_pre[i][:, col])

                for j in range(len(X_pre[i])):
                    if distance.euclidean(tmp_base, tmp_conv) > threshold:
                        to_drop.append(j)
            if len(to_drop) > 0:
                X_pre[i] = np.delete(X_pre[i], list(set(to_drop)), axis=0)

        # Resample in constant step (N=64)
        for i in range(len(X_pre)):
            X_pre[i] = utils.resample(X_pre[i], self.resampling_size)  # Resample and remove tailing points

        # Scale path-length to unity
        for i in range(len(X_pre)):
            # Compute sequence length
            length = 0.0
            for j in range(len(X_pre[i]) - 1):
                vec1 = X_pre[i][j, 0:3]
                vec2 = X_pre[i][j + 1, 0:3]
                length += np.linalg.norm(vec2 - vec1)
            X_pre[i][:, 0:3] /= length

        # Centering
        for i in range(len(X_pre)):
            mean = np.mean(X_pre[i][:, 0:3], axis=0)
            X_pre[i][:, 0:3] -= mean

        return X_pre

    def _feature_creation(self, X):
        # Feature computation
        curv_feat = curvature_feature(X)
        speed_feat = speed_feature(X)
        distance_feat = distance_feature(X)
        angle_feat = angle_feature(X)

        # Stack the created features
        X_svm = np.hstack([speed_feat, curv_feat, angle_feat, distance_feat])

        return X_svm  # Return matrix of new features


if __name__ == '__main__':
    # Data importing
    X, y = utils.read_files(1)

    # Hyper-parameters tuning
    thresholds = np.linspace(start=0.04, stop=0.12, num=5)
    best_score = 0.0
    best_params = ()
    for thresh in thresholds:
        cv = utils.LeaveOneOut()
        score = []
        for train_idx, val_idx in cv.split(X, y):
            model = SVM(C=1, gamma=1, kernel="rbf", threshold=thresh)
            model.fit(np.take(X, train_idx), np.take(y, train_idx))
            y_pred = model.predict(np.take(X, val_idx))
            score.append(utils.score(y_pred, np.take(y, val_idx)))
        score_mean = np.array(score).mean()
        if score_mean > best_score:
            best_score = score_mean
            best_params = thresh
            print("current best accuracy score={} with {}".format(best_score, best_params))
