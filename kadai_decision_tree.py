import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
X_train = []
y_train = []
y_pred = []
X_test = []
y_train_pred = []

path = 'train.dat'
path2 = 'test.dat'
with open(path) as f:
    for s_line in f:
        X_train.append(eval(s_line[2:]))
        y_train.append(int(s_line[0]))

with open(path2) as f2:
    for s_line2 in f2:
        X_test.append(eval(s_line2))

tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)

print(classification_report(y_train, y_train_pred))

y_pred = tree.predict(X_test)

with open('test.dat', mode='w') as f:
    for i in range(len(y_pred)):
        f.write(str(y_pred[i]) + ' ' + str(X_test[i]) + '\n')


