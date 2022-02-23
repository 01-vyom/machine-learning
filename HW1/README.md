# Homework-1
 
- 6 questions on basics of linear regression and regressive (multi-)class classification tasks.

- Helpful Reading: [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Chapter - 2, 3.5, 4.1-4.3 and [Pattern Recognition and
Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) Chapter - 2, 3.1, 4.1-4.2

- The last question uses the [20-Newsgroup Dataset](http://qwone.com/~jason/20Newsgroups/), where we have to perform classification of article topics using the given [matlab compatible dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz). Download and extract the dataset to work on. Cleaned the dataset by only taking words which have count greater than 1000 using [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) operations.

- Types of method used to solve the problem are described as follows:

1. Using Bernoulli Distribution for a word if it exists in a given document. Then performed [Bernoulli Naïve Bayes Classification](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_na%C3%AFve_Bayes) using [sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) implementation.

2. Using Multinomial Distribution for the frequency of words in the given document. Then performed [Multinomial Naïve Bayes Classification](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_na%C3%AFve_Bayes) using [sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) implementation.

3. Using multivariate normal distribution by assuming that their covariance matrices being the same, thus using a [Linear Discriminant Analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) classifier. For this, I used the [Term Frequency–Inverse Document Frequency features (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) features and then applied the LDA model using [sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) implementation.

4. Using multi-class least-square classifier with TF-IDF feature. For this, I used the [numpy package](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) implementation of least-squares problem by [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) the classes to perform argmax while predicting the labels.