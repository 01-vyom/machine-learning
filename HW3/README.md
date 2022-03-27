# Homework-3
 
- 4 questions on regularization, proximal gradient decent, subgradients, and subdifferential methods

- The 4th question uses the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/), where we have to build a a [multi-class support vector machine](https://en.wikipedia.org/wiki/Support-vector_machine) with [group sparse regularization](https://en.wikipedia.org/wiki/Structured_sparsity_regularization) using a [SGD algorithm](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). Download and extract the dataset to work on. We use the stochastic proximal subgradient algorithm for solving this problem. The algorithm is run on several different lambda values to see their effect. The sugradient algorithm is written in [numpy package](https://numpy.org/doc/stable/index.html). The algorithm is compared by checking how many images are accurately predicted.