
import matplotlib.pyplot as plt
import csv
import numpy as np

f = open('ds/namesex_data_v2.csv', 'r', encoding='utf8')
mydata = csv.DictReader(f)
sexlist = []
namelist = []
foldlist = []
for i, arow in enumerate(mydata):
    sexlist.append(int(arow['sex'].strip()))
    gname = arow['gname'].strip()
    namelist.append(gname)
    foldlist.append(int(arow['fold'].strip()))

sexlist = np.asarray(sexlist)
namelist = np.asarray(namelist)
foldlist = np.asarray(foldlist)
f.close()

# split the dataset according to folding number
train_idx = (foldlist <= 6)
valid_idx = (foldlist == 7)
stack_idx = (foldlist == 8)
test_idx  = (foldlist == 9)

# compute the appearances of each feature
name_feature, name_feature_cnt = [], []
for i in range(len(namelist[train_idx])):
    for j in range(len(namelist[train_idx][i])):
        if namelist[train_idx][i][j] not in name_feature:
            name_feature.append(namelist[train_idx][i][j])
            name_feature_cnt.append(1)
        else:
            idx = name_feature.index(namelist[train_idx][i][j])
            name_feature_cnt[idx] += 1
    if len(namelist[train_idx][i]) > 1:
        if namelist[train_idx][i] not in name_feature:
            name_feature.append(namelist[train_idx][i])
            name_feature_cnt.append(1)
        else:
            idx = name_feature.index(namelist[train_idx][i])
            name_feature_cnt[idx] += 1

# remove the feature with less than 2 appearances
i = 0
while i < len(name_feature):
    if name_feature_cnt[i] < 2:
        del name_feature[i]
        del name_feature_cnt[i]
        i -= 1
    i += 1

name_feature.append("_Other_Feature_")

x_train = np.zeros((len(namelist[train_idx]), len(name_feature)))
x_valid = np.zeros((len(namelist[valid_idx]), len(name_feature)))
x_stack = np.zeros((len(namelist[stack_idx]), len(name_feature)))
x_test  = np.zeros((len(namelist[test_idx]), len(name_feature)))

y_train = sexlist[train_idx]
y_valid = sexlist[valid_idx]
y_stack = sexlist[stack_idx]
y_test  = sexlist[test_idx]

# construct x_train
for i in range(len(namelist[train_idx])):
    other = 1
    for j in range(len(namelist[train_idx][i])):
        if namelist[train_idx][i][j] in name_feature:
            idx = name_feature.index(namelist[train_idx][i][j])
            x_train[i][idx] = 1
        else:
            other = 0
    if len(namelist[train_idx][i]) > 1:
        if namelist[train_idx][i] in name_feature:
            idx = name_feature.index(namelist[train_idx][i])
            x_train[i][idx] = 1
        else:
            other = 0
    if other == 0:
        x_train[i][-1] = 1

# construct x_valid
for i in range(len(namelist[valid_idx])):
    other = 1
    for j in range(len(namelist[valid_idx][i])):
        if namelist[valid_idx][i][j] in name_feature:
            idx = name_feature.index(namelist[valid_idx][i][j])
            x_valid[i][idx] = 1
        else:
            other = 0
    if len(namelist[valid_idx][i]) > 1:
        if namelist[valid_idx][i] in name_feature:
            idx = name_feature.index(namelist[valid_idx][i])
            x_valid[i][idx] = 1
        else:
            other = 0
    if other == 0:
        x_valid[i][-1] = 1

# construct x_stack
for i in range(len(namelist[stack_idx])):
    other = 1
    for j in range(len(namelist[stack_idx][i])):
        if namelist[stack_idx][i][j] in name_feature:
            idx = name_feature.index(namelist[stack_idx][i][j])
            x_stack[i][idx] = 1
        else:
            other = 0
    if len(namelist[stack_idx][i]) > 1:
        if namelist[stack_idx][i] in name_feature:
            idx = name_feature.index(namelist[stack_idx][i])
            x_stack[i][idx] = 1
        else:
            other = 0
    if other == 0:
        x_stack[i][-1] = 1

# construct x_test
for i in range(len(namelist[test_idx])):
    other = 1
    for j in range(len(namelist[test_idx][i])):
        if namelist[test_idx][i][j] in name_feature:
            idx = name_feature.index(namelist[test_idx][i][j])
            x_test[i][idx] = 1
        else:
            other = 0
    if len(namelist[test_idx][i]) > 1:
        if namelist[test_idx][i] in name_feature:
            idx = name_feature.index(namelist[test_idx][i])
            x_test[i][idx] = 1
        else:
            other = 0
    if other == 0:
        x_test[i][-1] = 1

print("x_train shape = ", x_train.shape)
print("y_train shape = ", y_train.shape)
print("x_valid shape = ", x_valid.shape)
print("y_valid shape = ", y_valid.shape)
print("x_stack shape = ", x_stack.shape)
print("y_stack shape = ", y_stack.shape)
print("x_test shape  = ", x_test.shape)
print("y_test shape  = ", y_test.shape)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
grid = np.zeros(15, dtype=int)
for i in range(len(grid)):
    if i == 0:
        grid[i] = 100
    else:
        grid[i] = grid[i-1]+100
for i in range(len(grid)):
    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=grid[i])
    clf.fit(x_train, y_train)
    ypred = clf.predict(x_valid)
    print("grid = "+str(grid[i])+", F1 score = "+str(f1_score(y_valid, ypred)))

