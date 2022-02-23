# -*- coding: utf-8 -*-
"""
@Author: Vyom Pathak
@Date: 02/18/2022
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import HuberRegressor
from sklearn import svm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

# Q5
data = pd.read_csv("winequality-red.csv", sep=";")

print(data.head())

print(data.info())

print(data.describe())

data_process = data.iloc[:, :11]
data_process["const"] = 1

train_data = data_process[:1400]
train_label = data["quality"][:1400]
test_data = data_process[1400:]
test_label = data["quality"][1400:]

print(train_data.head())

print(train_label.head())

print(test_data.head())

print(test_label.head())


def calc_metric(pred_label, label):
    print(
        "Mean Absolute Error on the Test Set is : ",
        mean_absolute_error(label, pred_label) * 100,
    )


# 1 - Least-Square Loss
W = np.linalg.lstsq(train_data, train_label, rcond=None)
pred_label = test_data @ W[0]
calc_metric(pred_label, test_label)

# 2 - Modified Huber Loss with M = 1
huber = HuberRegressor(epsilon=1, max_iter=215).fit(train_data, train_label)
pred_label = huber.predict(test_data)
calc_metric(pred_label, test_label)

# 3 - Hinge Loss
def loss_function(t):
    t = [max(0, np.abs(t) - 0.5) for t in t]
    return np.mean(t)


def objective_function(beta, X, Y):
    error = loss_function(X @ beta - Y)
    return error


beta_init = np.array([1] * train_data.shape[1])
result = minimize(
    objective_function,
    beta_init,
    args=(train_data, train_label),
    method="BFGS",
    options={"maxiter": 500},
)
W = result.x
pred_label = test_data.to_numpy() @ W
calc_metric(test_label, pred_label)

# Q6
data = pd.read_csv(
    "ionosphere.data", names=[i for i in range(34)] + ["label"], header=None,
)

print(data.head())

print(data.describe())

print(data.info())

data_process = data.iloc[:, :34]
data_process["const"] = 1
data["label"] = data["label"].apply(lambda x: 1 if x == "g" else -1)

train_data = data_process[:300]
train_label = data["label"][:300]
test_data = data_process[300:]
test_label = data["label"][300:]


def calc_accuracy_metric(pred_label, test_label):
    print(
        "Accuracy on the Test Set is : ", accuracy_score(pred_label, test_label) * 100
    )


# 1 - Modified Least Squares Loss
def loss_function(t, y):
    temp = t.T @ y
    return (temp - 1) ** 2


def objective_function(beta, X, Y):
    error = loss_function(X @ beta, Y)
    return error


beta_init = np.array([1] * train_data.shape[1])
result = minimize(
    objective_function,
    beta_init,
    args=(train_data, train_label),
    method="BFGS",
    options={"maxiter": 500},
)
W = result.x
pred_label = [1 if i > 0 else -1 for i in (test_data.to_numpy() @ W)]

calc_accuracy_metric(pred_label, test_label)

# 2 - Logistic loss function
clf = LogisticRegression(random_state=0).fit(train_data, train_label)
pred_label = clf.predict(test_data)
calc_accuracy_metric(test_label, pred_label)

# 3 - Hinge loss function
est = svm.LinearSVC(random_state=0, max_iter=300, loss="hinge", tol=1e-8)
est.fit(train_data, train_label)
pred_label = est.predict(test_data)
calc_accuracy_metric(pred_label, test_label)

