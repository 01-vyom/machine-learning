# -*- coding: utf-8 -*-
"""
@Author: Vyom Pathak
@Date: 04/20/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
import urllib.request
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

# Q3 - 20 News Group Revisited -
# a - LSI/PCA via orthogonal iteration
# b - GMM via EM


class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = bool(whiten)

    def fit(self, X):
        n, m = X.shape
        self.mu = X.mean(axis=0)
        X = X - self.mu
        C = X.T @ X / (n - 1)  # Eigen Decomposition
        C_k = C
        Q_k = np.eye(C.shape[1])
        for k in range(100):  # number of iterations=100
            # Q, R = QR(C_k)
            Q, R = np.linalg.qr(C_k)
            Q_k = Q_k @ Q
            C_k = R @ Q
        self.eigenvalues = np.diag(C_k)
        self.eigenvectors = Q_k
        if self.n_components is not None:  # truncate the number of components
            self.eigenvalues = self.eigenvalues[0 : self.n_components]
            self.eigenvectors = self.eigenvectors[:, 0 : self.n_components]
        descending_order = np.flip(
            np.argsort(self.eigenvalues)
        )  # eigenvalues in descending order
        self.eigenvalues = self.eigenvalues[descending_order]
        self.eigenvectors = self.eigenvectors[:, descending_order]
        return self

    def transform(self, X):
        X = X - self.mu
        if self.whiten:
            X = X / self.std
        return X @ self.eigenvectors


class GMM:
    def __init__(self, C, n_runs):
        self.C = C
        self.n_runs = n_runs

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, X, prediction):
        d = X.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)

        counter = 0
        for label in labels:
            ids = np.where(prediction == label)
            self.initial_pi[counter] = len(ids[0]) / X.shape[0]
            self.initial_means[counter, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.initial_means[counter, :]
            Nk = X[ids].shape[0]
            self.initial_cov[counter, :, :] = (
                np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            )
            counter += 1
        assert np.sum(self.initial_pi) == 1

        return (self.initial_means, self.initial_cov, self.initial_pi)

    def _initialise_parameters(self, X):
        n_clusters = self.C
        kmeans = KMeans(
            n_clusters=n_clusters, init="k-means++", max_iter=500, algorithm="auto"
        )
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        (
            self._initial_means,
            self._initial_cov,
            self._initial_pi,
        ) = self.calculate_mean_covariance(X, prediction)

        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self, X, pi, mu, sigma):
        N = X.shape[0]
        self.gamma = np.zeros((N, self.C))

        const_c = np.zeros(self.C)

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
            self.gamma[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    def _m_step(self, X, gamma):
        N = X.shape[0]
        C = self.gamma.shape[1]
        d = X.shape[1]

        self.pi = np.mean(self.gamma, axis=0)

        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis=0)[:, np.newaxis]

        for c in range(C):
            x = X - self.mu[c, :]

            gamma_diag = np.diag(self.gamma[:, c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c, :, :] = (sigma_c) / np.sum(self.gamma, axis=0)[:, np.newaxis][
                c
            ]

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, X, pi, mu, sigma):
        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(self.mu[c], self.sigma[c], allow_singular=True)
            self.loss[:, c] = self.gamma[:, c] * (
                np.log(self.pi[c] + 0.00001)
                + dist.logpdf(X)
                - np.log(self.gamma[:, c] + 0.000001)
            )
        self.loss = np.sum(self.loss)
        return self.loss

    def fit(self, X):
        d = X.shape[1]
        self.mu, self.sigma, self.pi = self._initialise_parameters(X)

        try:
            for run in range(self.n_runs):
                self.gamma = self._e_step(X, self.mu, self.pi, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)

        except Exception as e:
            print(e)

        return self

    def predict(self, X):

        labels = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            labels[:, c] = self.pi[c] * mvn.pdf(
                X, self.mu[c, :], self.sigma[c], allow_singular=True
            )
        labels = labels.argmax(1)
        return labels


newsgroups_train2 = fetch_20newsgroups(subset="train")  # with metadata
categories = newsgroups_train2.target_names
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=200, stop_words="english"
)
afterTFIDF = tfidf_vectorizer.fit_transform(newsgroups_train2.data)
data = afterTFIDF.toarray()

# a
pca = PCA(whiten=False, n_components=2)
pca.fit(data)
final = pca.transform(afterTFIDF.toarray())
print(final.shape)
colors = cm.rainbow(np.linspace(0, 1, len(categories)))
plt.figure(figsize=(7, 7))
plt.title("Principal Component Analysis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.scatter(final[:, 0], final[:, 1], c=colors[newsgroups_train2.target])
plt.legend(loc="best", markerscale=10)
plt.show()

# b
pca2 = PCA(whiten=False, n_components=100)
pca2.fit(data)
final1 = pca2.transform(data)
print(final1.shape)
gmm1 = GaussianMixture(
    n_components=20, covariance_type="full", max_iter=1000, verbose=1
)
gmm1.fit(final1)
means1 = gmm1.means_
y = final1 @ means1.T  # theta into muc
print(y.shape)
print("loading the dictionary")
url = "http://qwone.com/~jason/20Newsgroups/vocabulary.txt"
file = urllib.request.urlopen(url)

dict = {}
i = 0
for line in file:
    dict[i] = line.decode("utf-8").strip()
    i = i + 1
for i in range(20):
    temp = y[:, i]
    ind1 = temp.argsort()[-10:][::-1]
    print("\nTop 10 words for Cluster:", i + 1)
    top_10_words = []
    for i in ind1:
        top_10_words.append(dict[newsgroups_train2.target[i]])
    print(", ".join(top_10_words))

# Q4 Mixture of multinomials.
# c - Expectation-Maximization algorithm


class MultinomialMixture:
    def __init__(self, c, k):

        self.C = c
        self.K = k
        self.smoothing = 0.001

    def train(self, dataset, threshold=0, max_epochs=10):

        likelihood_list = []
        current_epoch = 1
        old_likelihood = -np.inf
        delta = np.inf

        # Initialisation of the model's parameters.
        # probility of each class
        pr = [1 / self.K] * self.C
        self.pi = pr

        self.p = np.empty((self.K, self.C))
        for i in range(0, self.C):
            em = np.random.uniform(size=self.K)
            em = em / np.sum(em)
            self.p[:, i] = em

        while current_epoch <= max_epochs and delta > threshold:
            # E-step
            posterior_estimate = np.divide(
                np.multiply(self.p[dataset, :], np.reshape(self.pi, (1, self.C))),
                np.dot(self.p[dataset, :], np.reshape(self.pi, (self.C, 1))),
            )
            # Compute the likelihood
            likelihood = np.sum(
                np.log(self.p[dataset, :] * np.reshape(self.pi, (1, self.C)))
            )
            likelihood_list.append(likelihood)
            # M-step
            self.pi = np.divide(
                self.smoothing + np.sum(posterior_estimate, axis=0),
                self.smoothing * self.C + np.sum(posterior_estimate),
            )
            self.p = np.divide(
                np.add.at(
                    self.smoothing + np.zeros((self.K, self.C)),
                    dataset,
                    posterior_estimate,
                ),
                np.reshape(
                    self.smoothing * self.K + np.sum(posterior_estimate[:, :], axis=0),
                    (1, self.C),
                ),
            )
            delta = likelihood - old_likelihood
            old_likelihood = likelihood
            current_epoch += 1
        return likelihood_list

    def predict(self, prediction_set):
        prods = self.p[prediction_set, :] * np.reshape(self.pi, (1, self.C))
        return np.argmax(prods, axis=1)

    def mean(self, size):
        mean = []
        for _ in range(0, size):
            state = np.random.choice(np.arange(0, self.C), p=self.pi)
            emitted_label = np.random.choice(np.arange(0, self.K), p=self.p[:, state])
            mean.append(emitted_label)
        return mean


# c
count_vectorizer = CountVectorizer()
aftercountvec = count_vectorizer.fit_transform(newsgroups_train2.data)
data = aftercountvec.toarray().T
print(data.shape)
data_target = newsgroups_train2.target
C = 11314
K = 20
dim_dataset = 130107
mixture = MultinomialMixture(C, K)
mixture.train(data, threshold=0.01)
mean = mixture.mean
theta_mean = np.matmul(data, mean.transpose())
for i in range(20):
    result = []
    temp = theta_mean[:, i]
    temp_index = temp.argsort()[-10:][::-1]
    print("\nTop 10 words for Cluster:", i + 1)
    top_10_words = []
    for i in temp_index:
        top_10_words.append(dict[data_target[i]])
    print(", ".join(top_10_words))
