import numpy as np
import pickle
from sklearn import preprocessing

class mylasso():
    def __init__(self, lamcoef = 0.1, max_iter=1000, tol=1e-6, const_regu = False):
        ### Add your code here ###
        self.lamcoef = lamcoef
        self.max_iter = max_iter
        self.tol = tol
        self.const_regu = const_regu
 
    def fit(self, x_train, y_train, winit = "ridge", keep_traindata = True, verbose = False):
        ### Add your code here ###
        if keep_traindata:
          self.x_train = x_train

        # calculate an initial w with Ridge Regression
        w = np.dot(np.linalg.inv(0.1 * np.identity(x_train.shape[1]) + np.dot(x_train.transpose(), x_train)), np.dot(x_train.transpose(), y_train))
        
        # calculate the loss function L or L'
        if self.const_regu:
            L = np.sum(np.power(y_train - np.dot(x_train, w), 2))/2/len(y_train) + self.lamcoef*np.sum(np.abs(w))
        else:
            L = np.sum(np.power(y_train - np.dot(x_train, w), 2))/2/len(y_train) + self.lamcoef*np.sum(np.abs(np.delete(w, 0, 0)))

        self.w = w
        L_min = L

        for i in range(self.max_iter):
            print("the " + str(i) + " iteration")      
            # implement coordinate descent with soft thresholding
            for j in range(len(w)):
                w_star = np.sum((y_train - np.dot(np.delete(x_train, j, 1), np.delete(w, j, 0))) * x_train[:, j]) / np.sum(np.dot(x_train[:, j].transpose(), x_train[:, j]))
                if w_star - len(y_train) * self.lamcoef / np.sum(np.dot(x_train[:, j].transpose(), x_train[:, j])) > 0:
                    w[j] = w_star - len(y_train) * self.lamcoef / np.sum(np.dot(x_train[:, j].transpose(), x_train[:, j]))
                elif w_star + len(y_train) * self.lamcoef / np.sum(np.dot(x_train[:, j].transpose(), x_train[:, j])) < 0:
                    w[j] = w_star + len(y_train) * self.lamcoef / np.sum(np.dot(x_train[:, j].transpose(), x_train[:, j]))
                else:
                    w[j] = 0
            
            # calculate the loss after an iteration
            if self.const_regu:
                L_new = np.sum(np.power(y_train - np.dot(x_train, w), 2))/2/len(y_train) + self.lamcoef*np.sum(np.abs(w))
            else:
                L_new = np.sum(np.power(y_train - np.dot(x_train, w), 2))/2/len(y_train) + self.lamcoef*np.sum(np.abs(np.delete(w, 0, 0)))

            # update w with the lowest loss
            if L_new < L_min:
                self.w = w
                L_min = L_new

            # compare the loss before and after an iteration
            if np.abs(L - L_new) < self.tol:
                break

            L = L_new
 
    def predict(self, x_test):
        ### Add your code here ###
        y_pred = np.dot(x_test, self.w)
        return y_pred


np.set_printoptions(suppress=True)
 
# Load data
with open('msd_data1.pickle', 'rb') as fh1:
    msd_data = pickle.load(fh1)

xscaler = preprocessing.StandardScaler().fit(msd_data['X_train'])
# standardize feature values
X_train_sd = xscaler.transform(msd_data['X_train'])
X_test_sd = xscaler.transform(msd_data['X_test'])

# add a column with all ones
const_train = np.ones((X_train_sd.shape[0], 1))
X_train_sd = np.concatenate((const_train, X_train_sd), axis=1)
const_test = np.ones((X_test_sd.shape[0], 1))
X_test_sd = np.concatenate((const_test, X_test_sd), axis=1)
 
# outcome values
Y_train = msd_data['Y_train']
Y_test = msd_data['Y_test']

mlo = mylasso(lamcoef = 0.1)
mlo.fit(X_train_sd, Y_train)
ypred = mlo.predict(X_test_sd)

RMSE = 0
for i in range(len(ypred)):
  RMSE += (ypred[i] - Y_test[i]) ** 2
RMSE = np.sqrt(RMSE / len(ypred))

print("RMSE = ", RMSE)