import numpy as np
import pandas as pd
!pip install mglearn

import mglearn

## Supervised Learning Basic

#### load dataset basic

from sklearn.datasets import load_iris
iris_dataset = load_iris()

iris_dataset

print("Keys of iris_dataset:\n", iris_dataset.keys())

print("Target names:", iris_dataset['target_names'])

print("Feature names:\n", iris_dataset['feature_names'])

print("Type of data:", type(iris_dataset['data']))

print("Shape of data:", iris_dataset['data'].shape)

print("First five rows of data:\n", iris_dataset['data'][:5])

print("Type of target:", type(iris_dataset['target']))

print("Shape of target:", iris_dataset['target'].shape)

print("Target:\n", iris_dataset['target'])

#### train/test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0) # 실행할때마다 다를 수 있음 그래서 random state를 지정해주면 고정됨 random적으로

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#### First Things First: Look at Your Data

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

#### Building Your First Model: k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

#### Making Predictions (example)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])

import matplotlib.pyplot as plt

plt.scatter(X_train[0], X_train[1])
plt.scatter(X_new, marker= '^')
plt.show()

#### Evaluating the Model

y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

y_pred == y_test

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

#### Summary

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

## Decision trees

##### DecisionTreeClassifier example

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# data load : breast_cancer dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# model training with labeled training data
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# prediction
y_train_hat = clf.predict(X_train)
y_test_hat = clf.predict(X_test)

# evaluation
from sklearn.metrics import accuracy_score
print("Accuracy on training set: ", accuracy_score(y_train, y_train_hat))
print("Accuracy on testing set: ", accuracy_score(y_test, y_test_hat))

#### varying the hyperparameter

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

training_accuracy = []
test_accuracy = []

msl_settings = [1, 2, 5, 7, 10, 20]
for msl in msl_settings:
    # model training with labeled training data
    clf = DecisionTreeClassifier(min_samples_leaf= msl, random_state=0)
    clf.fit(X_train, y_train)

    # prediction
    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    # evaluation
    training_accuracy.append(accuracy_score(y_train, y_train_hat))
    test_accuracy.append(accuracy_score(y_test, y_test_hat))

result = pd.DataFrame({"min_samples_leaf":msl_settings, "training accuracy": training_accuracy, "test accuracy": test_accuracy})

result

다른 hyperparameter도 바꿔가며 경향성을 확인해보세요.

#### Visualizing Decision Trees

clf = DecisionTreeClassifier(min_samples_leaf= 10, random_state=0)
clf.fit(X_train, y_train)

dt_clf_model_text = tree.export_text(clf)
print(dt_clf_model_text)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 8))
tree.plot_tree(clf,
                  feature_names=cancer.feature_names,
                  class_names=["malignant", "benign"],
                  filled=True)

Use 'graphviz' if you need more neat visualization.

from sklearn.tree import export_graphviz
import graphviz

export_graphviz(clf, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

#### Feature Importance in trees

print("Feature importances:")
print(clf.feature_importances_)

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plt.figure(figsize=(8,8))
plot_feature_importances_cancer(clf)

feature importance는 해당 feature의 방향성과 어떤 클래스를 지지하는지에 대한 정보는 제공하지 않음

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

