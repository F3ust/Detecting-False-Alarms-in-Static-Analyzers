import pandas as pd

import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_data = pd.read_csv("FA-detection-dataset/phoenix_train.csv")
test_data = pd.read_csv("FA-detection-dataset/phoenix_test.csv")

sns.countplot(x='label', data=train_data)
plt.title("Label(s) in train data")
plt.show()

target = "label"
x_train = train_data.drop(target, axis=1)
x_test = test_data.drop(target, axis=1)

dataframe = pd.DataFrame(data=x_train, columns=x_train.columns)
#dataframe["label"] = data["Outcome"]
matrix = dataframe.corr()

upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(np.bool_))
to_drop_1 = [column for column in upper.columns if any(upper[column] > 0.5)]
x_train.drop(to_drop_1, axis=1, inplace=True)
x_test.drop(to_drop_1, axis=1, inplace=True)

# dataframe = pd.DataFrame(data=train_data, columns=train_data.columns)
# matrix = dataframe.corr()
# cor_target = abs(matrix[target])
#
# irrelevant_features = cor_target[cor_target<0.4]
#
# cols = list([i for i in irrelevant_features.index])
#
# x_train = x_train.drop(cols, axis=1)
# x_test = x_test.drop(cols, axis=1)


dataframe = pd.DataFrame(data=x_train, columns=x_train.columns)
#dataframe["label"] = data["Outcome"]
matrix_n = dataframe.corr()

x_axis_labels = dataframe.columns
y_axis_labels = dataframe.columns

sns.heatmap(matrix_n, cmap="coolwarm", linewidths=.5, annot=True, annot_kws={"size":5}, vmin=-1, vmax=1, ax=None, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show()


y_train = train_data[target]
y_test = test_data[target]


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = RandomForestClassifier()

# param_grid = {
#     'n_estimators': [45, 50, 100, 200, 500],
#     'max_depth' : [None, 2,4,5,6,7,8],
#     'criterion' :['log_loss']
# }
#
# CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
# CV_clf.fit(x_train, y_train)
#
# print(CV_clf.best_params_)


#model = RandomForestClassifier(n_estimators=45, random_state=0, criterion="entropy", max_depth=None)
#model = SGDClassifier()
#model = LogisticRegression()
model = SVC()
#model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

f1 = f1_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
auc = roc_auc_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)


print("Precision score: {}".format(precision))
print("Recall score: {}".format(recall))
print("F1 score: {}".format(f1))
print("AUC score: {}".format(auc))
print(classification_report(y_test, y_predict))
# for i, j in zip(y_predict, y_test):
#     print("Prediction: {}. Actual Value: {}".format(i, j))



