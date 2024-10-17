# Low entropy --> High Information Gain
# High entropy --> Low Information Gain

from collections import Counter
import numpy as np




class Node:
  def __init__(self, feature = None,threshold = None,left = None,right = None,*,value=None):
   # decision-making point (tr = 0.5 in logistic regression)
   self.threshold = threshold
   # The feature used for splitting
   self.feature = feature
   # left tree
   self.left = left
   self.right = right
   # The class label if it's a leaf node (label = output variable we want to predict)
   self.value = value

  def is_leaf_node(self):
    return self.value is not None





class DecisionTree:
   def __init__(self, min_samples_split=2, max_depth = 100, n_features = None):
     self.min_samples_split = min_samples_split
     self.max_depth = max_depth
    # How many features should be used for each split in the decision tree
     self.n_features = n_features
     self.root = None

   def fit(self, X, y):
     self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features )
     self.root = self._grow_tree(X, y)



   def _grow_tree(self, X,y, depth=0):
     n_samples, n_feats = X.shape
     n_labels = len(np.unique(y))

    # Check the stopping criteria
     if (depth>= self.max_depth or n_labels == 1 or n_samples<self.min_samples_split):
       leaf_value = self._most_common_label(y)
       return Node(value= leaf_value)
     
    # Find the best split
    # Indices of the randomly selected n_features from n_feats - Once a feature is selected, it cannot be selected again (cannot have a selection like (f1,f1)) 
     feat_idxs = np.random.choice(n_feats, self.n_features, replace = False)
     best_feature,best_thresh = self._best_split(X, y, feat_idxs)

    # create child nodes
     left_idxs, right_idxs= self._split(X[:, best_feature], best_thresh)
     left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
     right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
     return Node(best_feature, best_thresh, left, right)
   

   # The most frequently occurring label in an array
   # e.g, 1/ y = ['Apple', 'Apple', 'Orange', 'Apple', 'Banana', 'Banana']
   def _most_common_label(self, y):
    # 2/ ({'Apple': 3, 'Banana': 2, 'Orange': 1})
    counter = Counter(y)
    # 3/ 'Apple'
    value = counter.most_common(1)[0][0]
    return value
     
   def _best_split(self,X, y, feat_idxs):
     best_gain = -1
     split_idx, split_threshold = None,None
     for feat_idx in feat_idxs:
      # All the rows from the column indicated by feat_idx
      X_column = X[:, feat_idx]
      # Check the thresholds_concept.txt
      thresholds = np.unique(X_column)

      # Goal of a decision tree algo is to find the threshold that maximizes the information Gain
      for thr in thresholds:
        # calculate the information gain
        gain = self._information_gain(y, X_column, thr)
        if gain > best_gain:
          best_gain = gain 
          split_idx = feat_idx
          split_threshold = thr
          
     return split_idx, split_threshold

   def _information_gain(self,y, X_column, threshold):
     # Create parent entropy - Before split
      parent_entropy = self._entropy(y)
     # Create children
      left_idxs, right_idxs = self._split(X_column, threshold)
      if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0
     #calculate the weighted avg. entropy of children
      n = len(y)
      n_l, n_r = len(left_idxs), len(right_idxs)
      e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[left_idxs])
      child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

      #calculate the IG
      information_gain = parent_entropy-child_entropy

      return information_gain
   

   def _entropy(self, y):
      # Number of occurrences
      # e.g, z = np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
      # z = array([1, 3, 1, 1, 0, 0, 0, 1])
      hist = np.bincount(y)
      # pi = ni/n (ni: nb of instances belonging to class i / n: the total nb of instances in y) 
      ps = hist / len(y)
      return -np.sum(p*np.log(p) for p in ps if p>0)
   

   def _split(self ,X_column, spit_thresh):
     # Indices of elements in X_column are less than or equal to the spit_thresh.
     left_idxs = np.argwhere(X_column <= spit_thresh).flatten()
     right_idxs = np.argwhere(X_column > spit_thresh).flatten()
     return left_idxs, right_idxs


   def predict(self,X):
     # X = sample_0 .. sample_n = x[0] .. x[n]
     return np.array([self._traverse_tree(x, self.root) for x in X])
   
   def _traverse_tree(self, x, node):
     if node.is_leaf_node():
       return node.value
     
     # node.feature is the index of split feature in X
     if x[node.feature] <= node.threshold :
       return self._traverse_tree(x, node.left)
     return self._traverse_tree(x, node.right)
     