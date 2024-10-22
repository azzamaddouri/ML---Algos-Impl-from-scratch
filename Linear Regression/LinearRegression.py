# y = weights X + bias
# dependent = slope * independent + intercept
# The best fit line using Gradient Descent
# Minimize cost function (Mean Square Error 3) using partial derivatives
# learning rate is how fast or slow to go in direction that gradient descent tells us to go

import numpy as np


class LinearRegression:
    def __init__(self, learningRate= 0.001,n_iters = 1000):
        self.learningRate = learningRate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # trainig
    def fit(self, X , y):
      n_samples, n_features = X.shape
      self.weights = np.zeros(n_features)
      self.bias = 0
      for _ in range(self.n_iters):
        y_pred = np.dot(X, self.weights) + self.bias
        dw = (1/n_samples) * np.dot(X.T, (y_pred -y))
        db = (1/n_samples) * np.sum(y_pred -y)
        self.weights = self.weights - self.learningRate * dw
        self.bias = self.bias - self.learningRate * db

    def predict(self, X):
       y_pred = np.dot(X,self.weights) + self.bias
       return y_pred