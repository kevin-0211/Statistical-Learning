import pickle
from sklearn import preprocessing
import numpy as np
from knn import myknn_regressor
from sklearn.neighbors import KNeighborsRegressor
import math
import matplotlib.pyplot as plt

def compute_rmse(ypred, Y_test):
	RMSE = 0
	for i in range(len(ypred)):
		RMSE += (ypred[i] - Y_test[i]) ** 2
	RMSE = math.sqrt(RMSE / len(ypred))
	return RMSE

#Load data
with open('msd_data1.pickle', 'rb') as fh1:
    msd_data = pickle.load(fh1)

xscaler = preprocessing.StandardScaler().fit(msd_data['X_train'])
#standardize feature values
X_train_sd = xscaler.transform(msd_data['X_train'])
X_test_sd = xscaler.transform(msd_data['X_test'])

X_train = msd_data['X_train']
X_test = msd_data['X_test']

Y_train = msd_data['Y_train']
Y_test = msd_data['Y_test']

k = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140, 160, 180, 200]
RMSE_1 = []
RMSE_2 = []
RMSE_3 = []

for i in range(len(k)):
	print(i)
	knn_1 = KNeighborsRegressor(n_neighbors=k[i])
	knn_1.fit(X_train_sd, Y_train)
	ypred_1 = knn_1.predict(X_test_sd)
	RMSE_1.append(compute_rmse(ypred_1, Y_test))

	knn_2 = KNeighborsRegressor(n_neighbors=k[i])
	knn_2.fit(X_train, Y_train)
	ypred_2 = knn_2.predict(X_test)
	RMSE_2.append(compute_rmse(ypred_2, Y_test))
	
	myknn = myknn_regressor(k[i], "remove_outliers")
	myknn.fit(X_train_sd, Y_train)
	ypred_3 = myknn.predict(X_test_sd)
	RMSE_3.append(compute_rmse(ypred_3, Y_test))
	

plt.figure(figsize=(20,10))
plt.plot(k, RMSE_1,'o-',color = 'r', label = 'first case')
plt.plot(k, RMSE_2,'o-',color = 'g', label = 'second case')
plt.plot(k, RMSE_3,'o-',color = 'b', label = 'third case')
plt.xlabel("k", fontsize=30, labelpad = 15)
plt.ylabel("RMSE", fontsize=30, labelpad = 15)
plt.savefig("tuning.png")