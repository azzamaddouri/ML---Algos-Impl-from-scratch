# ML algos are 3 catogories -> Supervised (Class label / Target variable) + Unsupervised (Set of features without target variable) + Reinforcement
# K Means = clustering algo + usupervised
# Elbow Technique to determine K

import numpy as np
import matplotlib.pyplot as plt

def euclidean_distances(x1, x2):
  return np.sqrt(np.sum((x2-x1)**2))
  
class KMeans:

 def __init__(self, K=5, max_iters=100, plot_steps=False):
   self.K = K
   self.max_iters = max_iters
   self.plot_steps = plot_steps

   # List of sample indices for each cluster
   self.clusters = [[] for _ in range(self.K)]

   # Centers (mean vector) for each cluster
   self.centroids = []

 def predict(self, X):
   self.X = X
   self.n_samples, self.n_features = X.shape

   # initialize
   random_sample_idxs= np.random.choice(self.n_samples, self.K, replace= False)
   self.centroids = [self.X[idx] for idx in random_sample_idxs]

   # optimisze clusters
   for _ in range(self.max_iters):
     
     # assign samples to closest centroids (create clusters)
     self.clusters = self._create_clusters(self.centroids)
     
     if self.plot_steps:
       self.plot()

     # calculate new centroids from the clusters
     centroids_old = self.centroids
     self.centroids = self._get_centroids(self.clusters)

     if self._is_converged(centroids_old, self.centroids):
       break
     
     if self.plot_steps:
       self.plot()
    
   # classify samples as the index of their clusters
   return self._get_cluster_labels(self.clusters)
 
 def _get_cluster_labels(self, clusters):
   # each sample will get the label of the cluster it was assigned to
   labels = np.empty(self.n_samples)
   # Access to Index and Value of each cluster
   for cluster_idx, cluster in enumerate(clusters):
    for sample_idx in cluster:
      # e.g, Samples 0 and 1 are part of Cluster 0, so they are labeled with 0.
      labels[sample_idx] = cluster_idx

   return labels

 def _create_clusters(self, centroids):
   # assign the samples to the closest centroids
   clusters = [[] for _ in range(self.K)]
   for idx, sample in enumerate(self.X):
    centroid_idx = self._closest_centroid(sample, centroids)
    clusters[centroid_idx].append(idx)
   return clusters

 def _closest_centroid(self,sample, centroids):
    # distance of the current sample to each centroid
    distances = [euclidean_distances(sample, point) for point in centroids]
    closest_idx = np.argmin(distances)
    return closest_idx
 

 def _get_centroids(self, clusters):
   # assign mean value of clusters to centroids
   # Centroids = mean position of all the data points in a cluster for each feature
   centroids = np.zeros((self.K, self.n_features))
   for cluster_idx, cluster in enumerate(clusters):
     # Select rows from self.X that correspond to the indices in cluster
     # axis=0 to calculate the mean of samples for each feature
     cluster_mean = np.mean(self.X[cluster], axis=0)
     centroids[cluster_idx] = cluster_mean
   return centroids
    
 
 def _is_converged(self, centroids_old ,centroids):
   # distances between old and new centroids
   distances = [euclidean_distances(centroids_old[i], centroids[i]) for i in range(self.K)]
   # Positions of centroids have not changed (they have converged)
   return sum(distances) == 0
 
 def plot(self):
        # create a figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # for each cluster scatter the points
        for i, index in enumerate(self.clusters):
            # Seperate x-coordinates from y-coordinates
            # (row1 = (x1,y1), row2 =(x2,y2) --> column1 = (x1,x2), column2= (y1,y2))
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

   
 
  
  
    



   





