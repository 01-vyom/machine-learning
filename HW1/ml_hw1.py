# -*- coding: utf-8 -*-
"""
@Author: Vyom Pathak
@Date: 02/03/2022
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Q1

phi = np.random.rand(40, 10)
psi = np.random.rand(40, 1)
theta = np.linalg.inv(phi.T @ phi) @ phi.T @ psi

for i in range(10):
    delta = np.random.rand(10, 1)
    part1 = np.linalg.norm(phi @ (theta + delta) - psi) ** 2
    part2 = np.linalg.norm(phi @ theta - psi) ** 2
    if part1 > part2:
        print(True)

# Q6

train_data_x = pd.read_csv(
    "20news-bydate/matlab/train.data", sep=" ", names=["doc_id", "word_id", "freq"],
)
# print(train_data_x.head())
train_data_y = pd.read_csv("20news-bydate/matlab/train.label", names=["labels"],)
# print(train_data_y.head())

test_data_x = pd.read_csv(
    "20news-bydate/matlab/test.data", sep=" ", names=["doc_id", "word_id", "freq"],
)
# print(test_data_x.head())
test_data_y = pd.read_csv("20news-bydate/matlab/test.label", names=["labels"],)
# print(test_data_y.head())

word_cnt = train_data_x[["word_id", "freq"]].groupby(["word_id"], as_index=False).sum()
vocab = list(word_cnt.loc[word_cnt["freq"] > 1000]["word_id"])

train_data_x_processed = train_data_x.loc[
    train_data_x["word_id"].isin(vocab)
].reset_index(drop=True)
train_data_processed = train_data_x_processed.join(
    train_data_y, how="inner", on="doc_id"
)
test_data_x_processed = test_data_x.loc[test_data_x["word_id"].isin(vocab)].reset_index(
    drop=True
)
test_data_processed = test_data_x_processed.join(test_data_y, how="inner", on="doc_id")

# Q6.a


def get_bernoulli_features(data, vocab):
    X = []
    Y = []
    for doc_id, doc in data.groupby(["doc_id"]):
        x = [0] * len(vocab)
        Y.append(list(doc["labels"])[0])
        doc_words = list(doc.word_id)
        for word_ind in range(len(vocab)):
            if vocab[word_ind] in doc_words:
                x[word_ind] = 1
        X.append(x)
    return np.array(X), np.array(Y)


X_train, Y_train = get_bernoulli_features(train_data_processed, vocab)
X_test, Y_test = get_bernoulli_features(test_data_processed, vocab)

clf = BernoulliNB()
clf.fit(X_train, Y_train)
print(clf.predict(X_test))
print(clf.score(X_test, Y_test) * 100)

# Q6.b


def get_multinomial_features(data, vocab):
    X = []
    Y = []
    for index, doc_group in data.groupby(["doc_id"]).agg(list).iterrows():
        x = [0] * len(vocab)
        Y.append(list(doc_group["labels"])[0])
        doc_words = list(doc_group.word_id)
        for word_ind in range(len(vocab)):
            if vocab[word_ind] in doc_words:
                x[word_ind] = doc_group.freq[doc_group.word_id.index(vocab[word_ind])]
        X.append(x)
    return np.array(X), np.array(Y)


X_train, Y_train = get_multinomial_features(train_data_processed, vocab)
X_test, Y_test = get_multinomial_features(test_data_processed, vocab)

clf = MultinomialNB()
clf.fit(X_train, Y_train)
print(clf.predict(X_test))
print(clf.score(X_test, Y_test) * 100)

# Q6.c


def get_tfidf_features(data, vocab):
    data = data.sort_values(by=["word_id"])
    vocab.sort()
    total_doc = len(data)
    word_group = data.groupby(["word_id"], as_index=False).agg(list)
    idf = []
    X = []
    Y = []
    for index, group in word_group.iterrows():
        idf.append(math.log(total_doc / len(group.doc_id)))
    for index, doc_group in data.groupby(["doc_id"]).agg(list).iterrows():
        x = [0] * len(vocab)
        Y.append(list(doc_group.labels)[0])
        for word_ind in range(len(vocab)):
            if vocab[word_ind] in doc_group.word_id:
                x[word_ind] = doc_group.freq[doc_group.word_id.index(vocab[word_ind])]
        X.append(x)
    tfidf = []
    for i in range(len(X)):
        x = []
        total_words_doc = sum(X[i])
        for j in range(len(X[0])):
            x.append(X[i][j] / total_words_doc * idf[j])
        tfidf.append(x)
    return np.array(tfidf), np.array(Y)


X_train, Y_train = get_tfidf_features(train_data_processed, vocab)
X_test, Y_test = get_tfidf_features(test_data_processed, vocab)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, Y_train)
print(clf.predict(X_test))
print(clf.score(X_test, Y_test) * 100)

# Q6.d


def get_least_square_features(data, vocab):
    X, Y = get_tfidf_features(data, vocab)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X, Y


X_train, Y_train = get_least_square_features(train_data_processed, vocab)
X_test, Y_test = get_least_square_features(test_data_processed, vocab)

# One-Hot Encoding Training Labels
Y_train_encoded = []
for y in Y_train:
    t = [0] * 20
    t[y - 1] = 1
    Y_train_encoded.append(t)

W = np.linalg.lstsq(X_train, Y_train_encoded, rcond=None)

Y_pred = np.argmax(X_test @ W[0], axis=1) + 1
print(Y_pred)
print(accuracy_score(Y_test, Y_pred) * 100)

