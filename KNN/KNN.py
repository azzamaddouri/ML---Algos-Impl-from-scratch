# Given data point :
# 1- calculate its distance from all other data points
# 2- Get the closest K points

from collections import Counter
import numpy as np

def euclidean_distances(x1, x2):
  return np.sqrt(np.sum((x2-x1)**2))

class KNN:

 def __init__(self, k=3):
   self.k=k

 def fit(self, X, y):
   self.X_train = X
   self.y_train = y

 def predict(self, X):
   predictions = [self._predict(x) for x in X]
   return predictions

 def _predict(self, x):
   # compute the distance
   distances = [euclidean_distances(x, x_train) for x_train in self.X_train]

   # get the closest k
   # Order distances and take the first k = 3 indices
   k_indices = np.argsort(distances)[:self.k]
   k_nearest_labels = [self.y_train[i] for i in k_indices]

   # majority vote
   most_common = Counter(k_nearest_labels).most_common()

   return most_common[0][0]