!pip install mglearn
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt



## RandomForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

plt.scatter(X[:,0], X[:,1], c= y)

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

forest.estimators_

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                                alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

training_accuracy = []
test_accuracy = []

n_settings = [1, 2, 5, 10, 20, 50, 100 ,1000]
for n in n_settings:
    clf = RandomForestClassifier(n_estimators=n)
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    training_accuracy.append(accuracy_score(y_train, y_train_hat))
    test_accuracy.append(accuracy_score(y_test, y_test_hat))

pd.DataFrame({"n_estimators":n_settings, "training accuracy": training_accuracy, "test accuracy": test_accuracy})

#### theoretically, more trees lead to stable model.

#### how about scaling ? 스케일링 할 필요 없음
#### DT를 기본적인 모델로 사용하기 때문에


### feature importance

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

clf.feature_importances_

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(clf)

## Adaboost

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=5, random_state=42)
ada.fit(X_train, y_train)

ada.estimators_

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), ada.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(ada, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("AdaBoost")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()

## Bagging (9 logistic regression models + voting)

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

#### training

n_estimators=9

logregs=[]
for i in range(n_estimators):
    ''''TODO - bootstrap sampling (You can use 'np.random.choice')'''
    np.random.seed(i)
    idx = np.random.choice(X_train.shape[0], X_train.shape[0], replace = True) # X_train.shape[0] -> 샘플의 갯수 --> 샘플의 갯수중에 샘플의 갯수만큼뽑겠다
                                                                               # replace -> 복원 추출
    X_train_base = X_train[idx,:]
    y_train_base = y_train[idx]

    logregs.append(LogisticRegression().fit(X_train_base, y_train_base))

logregs

#### test

y_test_hats=[]
for i in range(n_estimators):
    '''TODO - get y_test_hat of each base model'''
    y_test_hat = logregs[i].predict(X_test)
    y_test_hats.append(y_test_hat)

y_test_hats = np.stack(y_test_hats).T

y_test_hats

from scipy import stats
'''TODO - use 'stats.mode' function to get the result of majority voting

'''

accuracy_score(y_test, y_test_hat_voted)

# accuracy of each logistic regression model
for i in range(n_estimators):
    print(accuracy_score(y_test, y_test_hats[:,i]))
