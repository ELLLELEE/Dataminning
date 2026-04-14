import sklearn

print(sklearn.__version__)

!pip install mglearn
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

## Linear Models

#### Linear regression aka ordinary least squares

X, y = mglearn.datasets.make_wave(n_samples=60)

plt.scatter(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

# check the learned parameters
print("lr.coef_:", lr.coef_) # this is w1
print("lr.intercept_:", lr.intercept_) # this is w0 = b

# y= -0.0318 + 0.3939x
plt.scatter(X, y)
x_points = range(-3,4)
plt.plot(x_points, lr.coef_*x_points + lr.intercept_, c='red' )

y_train_hat = lr.predict(X_train)


y_test_hat = lr.predict(X_test)
print(y_test)
print(y_test_hat)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('performance for TRAIN--------')
print('train MAE : ', mean_absolute_error(y_train, y_train_hat))
print('train RMSE : ', mean_squared_error(y_train, y_train_hat)**0.5)
print('train R_square : ', r2_score(y_train, y_train_hat))

print('performance for TEST--------')
print('test MAE : ', mean_absolute_error(y_test, y_test_hat))
print('test RMSE : ', mean_squared_error(y_test, y_test_hat)**0.5)
print('test R_square : ', r2_score(y_test, y_test_hat))

#### linear regression on extended_boston dataset

- The dataset consists of 506 data points described by 104 features
- The 104 features are the 13 original features together with the 91 possible combinations of two features within those 13 (all products between original features).
- The regression task associated with this dataset is to predict the median value of homes in several Boston neighborhoods in the 1970s, using information such as crime rate, proximity to the Charles River, highway accessibility, and so on.

X, y = mglearn.datasets.load_extended_boston()
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train) # same as LinearRegression().fit(X_train, y_train)

y_train_hat = lr.predict(X_train)
y_test_hat = lr.predict(X_test)

print('performance for TRAIN--------')
print('train MAE : ', mean_absolute_error(y_train, y_train_hat))
print('train RMSE : ', mean_squared_error(y_train, y_train_hat)**0.5)
print('train R_square : ', r2_score(y_train, y_train_hat))

print('performance for TEST--------')
print('test MAE : ', mean_absolute_error(y_test, y_test_hat))
print('test RMSE : ', mean_squared_error(y_test, y_test_hat)**0.5)
print('test R_square : ', r2_score(y_test, y_test_hat))

Is this result good enough? overfitting or underfitting?

overfitting -> error가 train데이터에서 더 많이 작고 상관계수인 R_square이 더 큼


##### Ridge regression

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

y_train_hat = ridge.predict(X_train)
y_test_hat = ridge.predict(X_test)

print('performance for TRAIN--------')
print('train MAE : ', mean_absolute_error(y_train, y_train_hat))
print('train RMSE : ', mean_squared_error(y_train, y_train_hat)**0.5)
print('train R_square : ', r2_score(y_train, y_train_hat))

print('performance for TEST--------')
print('test MAE : ', mean_absolute_error(y_test, y_test_hat))
print('test RMSE : ', mean_squared_error(y_test, y_test_hat)**0.5)
print('test R_square : ', r2_score(y_test, y_test_hat))

Compare it with the LinearRegression results. Which one is better?

#### varying the hyperparameter

training_r2 = []
test_r2 = []

alpha_settings = [0, 0.1, 1, 10]
for alpha in alpha_settings:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    y_train_hat = ridge.predict(X_train)
    training_r2.append(r2_score(y_train, y_train_hat))

    y_test_hat = ridge.predict(X_test)
    test_r2.append(r2_score(y_test, y_test_hat))

pd.DataFrame({"alpha":alpha_settings, "training R2": training_r2, "test R2": test_r2})

#### effect of the hyperparameter alpha on RidgeRegression

ridge = Ridge(alpha=1).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()

##### Lasso

from sklearn.linear_model import Lasso

training_r2 = []
test_r2 = []
num_vars_used = []

alpha_settings = [0.0001, 0.001, 0.01, 0.1, 1]
for alpha in alpha_settings:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    num_vars_used.append(sum(lasso.coef_ != 0))

    y_train_hat = lasso.predict(X_train)
    training_r2.append(r2_score(y_train, y_train_hat))

    y_test_hat = lasso.predict(X_test)
    test_r2.append(r2_score(y_test, y_test_hat))

pd.DataFrame({"alpha":alpha_settings, "training R2": training_r2, "test R2": test_r2, "variables used": num_vars_used})

lasso = Lasso().fit(X_train, y_train)
lasso001 = Lasso(alpha=0.01, max_iter=1000).fit(X_train, y_train)
lasso00001 = Lasso(alpha=0.0001, max_iter=1000).fit(X_train, y_train)

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

##### Logistic Regression

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

training_accuracy = []
test_accuracy = []

C_settings = [0.01, 0.1, 1, 10, 100, 1000, 10000]
for C in C_settings:
    '''TODO'''
    lr = LogisticRegression(C=C)
    lr.fit(X_train,y_train)

    y_train_hat= lr.predict(X_train)
    training_accuracy.append(r2_score(y_train, y_train_hat))

    y_test_hat = lr.predict(X_test)
    test_accuracy.append(r2_score(y_test, y_test_hat))








pd.DataFrame({"C":C_settings, "training accuracy": training_accuracy, "test accuracy": test_accuracy})
# C가 커지면커질수록 overfitting

logreg = LogisticRegression().fit(X_train, y_train)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()

**Strengths**
- Linear models are very fast to train, and also fast to predict.
- They scale to very large datasets and work well with sparse data.
- They make it relatively easy to understand how a prediction is made.

**Weaknesses**
- If your dataset has highly correlated features, it is often not entirely clear why coefficients are the way they are. (It is important to remove redundant features –feature selection)
- They would perform worse if the relationship between features and target in your dataset is non-linear.
