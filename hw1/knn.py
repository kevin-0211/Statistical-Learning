import pickle
from sklearn import preprocessing
import numpy as np

class myknn_regressor():
    def __init__(self, n_neighbors = 10, mean_type = "equal_weight"):
        # initialize parameters
        self.n_neighbors = n_neighbors
        if n_neighbors < 10:
            self.mean_type = "equal_weight"
        else:
            self.mean_type = mean_type

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

    def predict(self, x_test):
        y_pred = np.zeros(len(x_test))
        for i in range(len(x_test)):
            print(i)
            # calculate the distance
            dist = np.zeros(len(self.x))
            for j in range(len(self.x)):
                dist[j] = np.sum((x_test[i] - self.x[j]) ** 2)

            # find k nearnest neighbors
            min_dist = np.sort(dist)
            min_dist_list = list(map(list(dist).index, min_dist[:self.n_neighbors]))

            # find out y according to the values of k nearnest neighbors' distances
            y_a = np.zeros(len(min_dist_list))
            for j in range(len(min_dist_list)):
                y_a[j] = self.y[min_dist_list[j]]

            # remove outliers
            if self.mean_type == "remove_outliers":
                Q1 = np.quantile(y_a, 0.25)
                Q3 = np.quantile(y_a, 0.75)
                IQR = Q3 - Q1
                y_a = y_a[(y_a >= (Q1 - 1.5 * IQR)) & (y_a <= (Q3 + 1.5 * IQR))]

            # calculate the mean
            y_pred[i] = np.sum(y_a) / len(y_a)
        return y_pred

#Load data
with open('msd_data1.pickle', 'rb') as fh1:
    msd_data = pickle.load(fh1)

doscaling = 1

if (doscaling == 1):
    xscaler = preprocessing.StandardScaler().fit(msd_data['X_train'])
    #standardize feature values
    X_train = xscaler.transform(msd_data['X_train'])
    X_test = xscaler.transform(msd_data['X_test'])
else:
    X_train = msd_data['X_train']
    X_test = msd_data['X_test']

Y_train = msd_data['Y_train']
Y_test = msd_data['Y_test']

myknn = myknn_regressor(20, "remove_outliers")
myknn.fit(X_train, Y_train)
ypred = myknn.predict(X_test)

RMSE = 0
for i in range(len(ypred)):
	RMSE += (ypred[i] - Y_test[i]) ** 2
RMSE = np.sqrt(RMSE / len(ypred))

print("RMSE = ", RMSE)
print("first 20 prediction = \n", ypred[:20])