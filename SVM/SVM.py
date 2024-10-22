# SVM use a linear model (wx+b) and try to find a linear decision boundary
# Maximize margin between the nearby data points and the line that seperates the clusters
# Nearby data points are called Support Vectors
# 2 features the boundary a line
# 3 features the boundary a plane
# n features the boundary a hyperplane
# loss function measures the error for a single individual instance
# cost function is the average of the loss function over the entire dataset
# Hinge Loss type of loss function --> L(y,f(x))=max(0,1−y⋅f(x))
# Gamma 
# Regularization

import numpy as np

class SVM:

 def __init__(self, learning_rate= 0.001, lambda_param= 0.01, n_iters=1000):
   self.lr = learning_rate
   self.lambda_param = lambda_param
   self.n_iters = n_iters
   self.w = None
   self.b = None

 def fit(self, X, y):
   n_samples, n_faetures = X.shape

   # Convert target labels to be (-1 and 1) rather than (0 , 1)
   y_ = np.where(y <=0,-1,1)

   self.w = np.zeros(n_faetures)
   self.b = 0
   
   for _ in range(self.n_iters):
     for idx, x_i in enumerate(X):
       # if true max loss function = 0
       condition = y_[idx] * (np.dot(x_i, self.w)- self.b) >= 1
       if condition:
         self.w -= self.lr * (2 * self.lambda_param * self.w)
       else:
          self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
          self.b -= self.lr * y_[idx]
      
    

 def predict(self, X):
   approx = np.dot(X, self.w) - self.b
   # return -1 or 1
   return np.sign(approx)
