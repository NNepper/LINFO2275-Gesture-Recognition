import numpy as np


class DTW():

    n_neighbors = np.Inf
    X = None
    labels = None
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        
    
    def fit(self, X, y):
        self.X = X
        self.labels = y


    # Dynamic Time Warping distance between sequence S1 and S2
    def _distance(self, s1, s2):
        dtw = np.zeros(shape=(len(s1),len(s2)))

        for i in range(0, len(s1)):
            for j in range(0, len(s2)):
                dtw[i, j] = np.Inf

        dtw[0, 0] = 0

        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                cost = np.linalg.norm(s1[i,:3] - s2[j,:3])
                dtw[i,j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        return dtw[-1, -1]

    def _distance_matrix(self, ms1, ms2):
        D = np.zeros((len(ms1), len(ms2)))      # TODO: Optimize memory usage by storing only upper triangle
        
        max_count = len(ms1) * len(ms2)
        count = 0
        for i in range(len(ms1)):
            for j in range(i + 1, len(ms2)):
                D[i,j] = self._distance(ms1[i], ms2[j])
                count += 1
                print("{}/{}".format(count, max_count))
        return D

    def predict(self, X):
        # Compute distance between train and test
        D = self._distance_matrix(self.X, X)
        sorted_idx = np.unravel_index((-D).argsort()[:self.n_neighbors], D.shape)
        sorted_labels = [self.labels[i] for (i,_) in sorted_idx]
        
        print(sorted_labels)




# Read specific filename from specified domain
def read_files(domain):
    
    X = []
    y = []

    for i in range(1, 1000):
        path = "data/Domain0{}/{}.txt".format(domain, i)

        with open(path, "r") as file:
            lines = file.readlines()

            # Store class
            clas = int(lines[1].split()[-1].rstrip("\n")) -1
            y.append(clas)

            # Store Sequence
            tmp = []
            for i in range(5, len(lines)):
                tmp.append([float(val) for val in lines[i].rstrip("\n").split(",")])
            X.append(np.matrix(tmp))
    return X,y
    

# Resample the measurement to standardized batch
def resample(sample, new_size):
    step = len(sample) / (new_size-1)
    D = 0
    pt1 = sample[0, 0:3]

    for i in range(1, 2):
        pt2 = sample[i,0:3]
        dist = np.linalg.norm(pt1 - pt2)
        
        if D+dist > step:
            pt_mid = pt1 + ((step - D)/dist) * pt_1
        else:
            D += dist
        pt1 = pt2


X,y =read_files(1)

X_train = X[:50]
X_test = X[50:]
y_train = y[50:60]
y_test = y[50:60]

model = DTW(5)
model.fit(X_train, y_train)
model.predict(X_test)