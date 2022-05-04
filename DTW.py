import numpy as np


class DTW():

    n_neighbors = np.Inf
    X = None
    labels = None
    window = np.Inf
    
    def __init__(self, n_neighbors=5, window_size=100):
        self.n_neighbors=n_neighbors
        self.window = window_size
        
    
    def fit(self, X, y):
        self.X = X
        self.labels = y


    # Dynamic Time Warping distance between sequence S1 and S2
    def _distance(self, s1, s2):
        dtw = np.zeros(shape=(len(s1),len(s2)))
        w = max(self.window, abs(len(s1) - len(s2)))

        for i in range(0, len(s1)):
            for j in range(max(1, i-w), min(len(s2), i+w)):
                dtw[i, j] = np.Inf

        dtw[0, 0] = 0

        for i in range(1, len(s1)):
            for j in range(max(1, i-w), min(len(s2), i+w)):
                cost = np.linalg.norm(s1[i,:3] - s2[j,:3])
                dtw[i,j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        return dtw[-1, -1]

    def _distance_matrix(self, ms1, ms2):
        D = np.zeros((len(ms1), len(ms2)))
        
        max_count = len(ms1) * len(ms2)
        count = 0
        for i in range(len(ms1)):
            for j in range(len(ms2)):
                D[i,j] = self._distance(ms1[i], ms2[j])
                count += 1
                if (count % 1000 == 0):
                    print("{}/{}".format(count, max_count))
        return D

    def predict(self, X):
        # Compute distance between train and test
        D = self._distance_matrix(self.X, X)
        sorted_idx = (-D).argsort(axis=0)[:self.n_neighbors]

        # label retrieval
        sorted_labels = np.array([[self.labels[sorted_idx[j,i]] for i in range(len(X))] for j in range(self.n_neighbors)])
        vote = [np.bincount(sorted_labels[:, i]) for i in range(len(X))]
        return [np.argmax(vote[i]) for i in range(len(X))]


# Read specific filename from specified domain
def read_files(domain):
    
    X = []
    y = []

    for i in range(1, 1001):
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
def train_test_split(X, y, train_ratio):
    cutting_pt = int(train_ratio*10)
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(0, 1000, 10):
        sampling = np.arange(10)
        np.random.shuffle(sampling)

        for j in range(0,cutting_pt):
            X_train.append(X[i+sampling[j]])
            y_train.append(y[i+sampling[j]])
        for j in range(cutting_pt,10):
            X_test.append(X[i + sampling[j]])
            y_test.append(y[i + sampling[j]])
    return np.array(X_train, dtype=object), np.array(y_train, dtype=object), np.array(X_test, dtype=object), np.array(y_test, dtype=object)


X,y =read_files(1)

X_train, y_train, X_test, y_test = train_test_split(X, y, 0.7)
print(X_train.shape)

print("training data class:", y_train)

window_size = 5
print(len(X_train), len(X_test))

model = DTW(3, window_size)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("prediction:", pred)
print("vs")
print("real values:", y_test)