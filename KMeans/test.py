import numpy as np
from sklearn.datasets import make_blobs
from KMeans import KMeans 

# Testing
if __name__ == "__main__":
    
    # Random numbers generated are the same every time you run the code
    np.random.seed(42)
    

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()