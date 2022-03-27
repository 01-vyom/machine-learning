# -*- coding: utf-8 -*-
"""
@Author: Vyom Pathak
@Date: 03/26/2022
"""

# https://www.kaggle.com/code/boltcoder/ml-hw3

# Q4

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Train Set: ", x_train.shape, y_test.shape)
print("Test Set: ", x_test.shape, y_test.shape)

print("Train Image Samples")
plt.subplots(figsize=(20, 20))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    num = random.randint(0, x_train.shape[0])
    plt.imshow(x_train[num], cmap="gray")
    plt.title("Digit = {}".format(y_train[num]))
plt.show()

print("Test Image Samples")
plt.subplots(figsize=(20, 20))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    num = random.randint(0, x_test.shape[0])
    plt.imshow(x_test[num], cmap="gray")
    plt.title("Digit = {}".format(y_test[num]))
plt.show()

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_train_flat = x_train_flat.astype(float)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
x_test_flat = x_test_flat.astype(float)
mean_image = np.mean(x_train_flat, axis=0)
x_train_flat -= mean_image
x_test_flat -= mean_image
print("train_set_x_flatten shape: " + str(x_train_flat.shape))
print("test_set_x_flatten shape: " + str(x_test_flat.shape))


def block_soft_threshold(W, mu):
    for i in range(0, W.shape[0]):
        l2norm = np.linalg.norm(W[i][:])
        if l2norm == 0:
            continue
        W[i][:] = np.multiply(W[i][:] / l2norm, np.maximum(l2norm - mu, 0))
    return W


def svm_naive(W, X, y, reg, i):
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # Compute the loss and the gradient.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    # Iterate over incorrect classes.
    for j in range(num_classes):
        if j == y[i]:
            continue
        # If margin is met, accumulate loss for the jth example
        # and calculate associated gradient.
        margin = scores[j] - correct_class_score + 1
        if margin > 0:
            dW[:, j] += X[i, :]
            dW[:, y[i]] -= X[i, :]
    return dW


def error_calculate(W):
    error = 0
    for i in range(0, len(x_test_flat)):
        x_i = x_test_flat[i]
        max_val = -1
        max_i = 10
        for digit in range(0, 10):
            res = np.dot(x_i.T, W[:, digit])
            if res > max_val:
                max_val = res
                max_i = digit
        if max_i != y_test[i]:
            error += 1
    return 100 * error / len(x_test_flat)


dimR = 784
dimC = 10
max_iter = 10 ** 6
lmda = [10, 1, 0.1, 0.01]
y_train = np.array(y_train)
y_train_flat = y_train.reshape(y_train.shape[0], -1)
data = [[] for y in range(10)]
for index in range(0, len(y_train_flat)):
    val = y_train_flat[index][0]
    data[val].append(index)
l = lmda[0]


error_all = []
W_all = []
for l in lmda:
    print("Training started for lambda =", l)
    W = np.zeros((dimR, dimC))
    dW = np.zeros(W.shape)
    error_PG = []

    for t in range(1, max_iter + 1):
        for i in range(0, 10):
            rand_index = data[i][random.randrange(len(data[i]))]
            dW = svm_naive(W, x_train_flat, y_train, 1, rand_index)
            W = W - (1 / t) * dW
        W = block_soft_threshold(W, l / t)
        if t % 1000 == 0 or t == 1:
            error = error_calculate(W)
            error_PG.append(error)
        if t % 100000 == 0 or t == 1:
            print("iter= {}, Accuracy= {:3f}".format(t, 100 - error))
    W_all.append(W)
    error_all.append(np.array(error_PG))

for i in range(len(error_all)):
    error_PG = error_all[i]
    t = np.arange(0, 1001)
    plt.semilogy(
        t, 100 - np.array(error_PG), linewidth=1, label="lambda = " + str(lmda[i])
    )
plt.legend()
plt.title("Proximal Stochiastic Gradient")
plt.xlabel("Iteration (10^3)")
plt.ylabel("Prediction accuracy percentage")
plt.show()
plt.subplots(figsize=(15, 15))
for k in range(len(W_all)):
    W = W_all[k]
    num = random.randint(0, 9)
    pixels = W[:, num] > 0
    pixels = pixels.reshape((28, 28))
    plt.subplot(1, len(W_all), k + 1)
    plt.title("Digit = {}, lambda is {}".format(num, lmda[k]))
    plt.imshow(pixels, cmap="gray")
    count_zeros = 0
    for i in np.sum(W, axis=1) / 10:
        if i == 0:
            count_zeros += 1
    print(
        "Total No of features being discarded (Zeros) for lambda ",
        lmda[k],
        ": ",
        count_zeros,
    )
plt.show()

