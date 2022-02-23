# Homework-2
 
- 6 questions on convex optimization problems

- Helpful Reading: [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)

- The 5th question uses the [Wine Quality Dataset](http://archive.ics.uci.edu/ml/datasets/Wine+Quality), where we have to build a linear model of the first 11 features. Download and extract the dataset to work on. Calculate the [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) using the [Sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).

- Types of loss-functions to solve the optimization problems are described as follows:

1. Using least-square loss function. I used the [numpy package](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) implementation of least-squares problem.

2. Using [Huber Loss function](https://en.wikipedia.org/wiki/Huber_loss). Here, we set the delta/M value as 1. Using [sklearn package](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html) implementation for huber regression.

3. Using a modified Hinge Loss described in the question. As the hinge loss is modified I buit it as a python3 function and passed it as a criteria argument to the [scipy package function (minimize)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) to solve the problem.

- The 6th question uses the [Ionosphere Dataset](https://archive.ics.uci.edu/ml/datasets/ionosphere), where we have to build a classification model of the 34 features to predict the binary outcome (±1). Download and extract the dataset to work on. Calculate the [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) using the [Sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).

- Types of loss-functions to solve the optimization problems are described as follows:

1. Using a modified least-square loss function. As the least-squares loss is modified I buit it as a python3 function and passed it as a criteria argument to the [scipy package function (minimize)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) to solve the problem.

2. Using [Logistic-Loss function](https://en.wikipedia.org/wiki/Logistic_regression). Here I used the [Sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to solve the logistic-loss problem.

3. Using the [Hinge loss function](https://en.wikipedia.org/wiki/Hinge_loss). I used the [Sklearn package function (Linear SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) to solve the Hinge loss problem.