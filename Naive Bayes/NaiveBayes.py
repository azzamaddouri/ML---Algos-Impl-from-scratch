# Bayes theorem + Naive assumption that n features aren't going to be independent when they may not be 
# posterior probability = likelihood * prior probability (What you think before seeing the data) / marginal likelihood
# Maximizing posterior probability helps find the most likely estimate of a parameter by combining prior beliefs and new data
# posterior probability = Model with Gaussian * Frequency of each class

import numpy as np


class NaiveBayes:

 def fit(self, X,y):
  n_samples, n_features = X.shape
  self._classes = np.unique(y)
  n_classes = len(self._classes)

  # calculate mean, var and prior for each class
  self._mean = np.zeros((n_classes,n_features), dtype=np.float64)
  self._variance = np.zeros((n_classes,n_features), dtype=np.float64)
  self._priors = np.zeros(n_classes, dtype=np.float64)

  for idx, c in enumerate(self._classes):
    X_c = X[y==c]
    self._mean[idx, : ] = X_c.mean(axis=0)
    self._variance[idx, : ] = X_c.var(axis=0)
    self._priors[idx] = X_c.shape[0] / float(n_samples)

 def predict(self, X):
   y_pred = [self._predict(x) for x in X]
   return np.array(y_pred)
 

 def _predict(self, x):
  posteriors = []

  # calculate posteriors probability for each class
  for idx, c in enumerate(self._classes):
   prior = np.log(self._priors[idx])
   posterior = np.sum(np.log(self._pdf(idx,x)))
   posterior = posterior + prior
   posteriors.append(posterior)

   # return class with the highest posterior
  return self._classes[np.argmax(posteriors)]
  

  # Probability of density functin
 def _pdf(self, class_idx, x):
  mean = self._mean[class_idx]
  var = self._variance[class_idx]
  numerator = np.exp(-((x-mean) ** 2) / (2 * var) )
  denominator = np.sqrt(2* np.pi * var)
  return numerator / denominator

    





