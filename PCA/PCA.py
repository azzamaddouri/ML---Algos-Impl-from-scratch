# PCA is the process of figuring out the most important features that has the most impact on target variable
# goal is maximize the variance (spread) and minimize the projection error

import numpy as np

class PCA:

 def __init__(self, n_components):
   # number of features after the transformation we want to have
   self.n_components = n_components
   self.mean = None

 def fit(self, X):
   # Calculate the mean of each feature for mean centering (mean of 0)
   # mean of 0 = the mean of the centered feature (xi-meani) = 0
   self.mean = np.mean(X, axis =0)
   X = X- self.mean

   # covariance , function needs samples as columns
   # Covariance provides a measure of how much two random variables change together
   # covariance > 0 : one variable increases, the other tends to increase as well // covariance < 0 one variable increases, the other tends to decrease
   cov = np.cov(X.T)

   # eigenvectors, eigenvalues for each feature
   # data projected onto eigenvectors, it creates a new coordinate system where the axes (PCA) are aligned with the directions of maximum variance
   # square matrix * eigenvector = eingenvalue * eigenvalue (A * v = lambda * v )
   eigenvectors, eigenvalues = np.linalg.eig(cov)

   # eigenvectors v = [:, i] column vector , transpose this for easier calculations
   eigenvectors = eigenvectors.T


   # sort eigenvectors in descending order
   idxs = np.argsort(eigenvalues)[::-1]
   eigenvalues = eigenvalues[idxs]
   eigenvectors = eigenvectors[idxs]

   # select the n_components eigenvector (the n_components corresponding to the largest eigenvalue)
   self.components = eigenvectors[: self.n_components]

 def transform(self, X):
   # project data
   X= X-self.mean
   # transpose for a row vector
   return np.dot(X, self.components.T)
  