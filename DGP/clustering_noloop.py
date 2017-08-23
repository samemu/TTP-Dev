import numpy as np
from sklearn.cluster import KMeans


'''README:
        > K-means clustering is non-deterministic. It chooses cluster centers 
        randomly and goes through iterations, choosing the model with the 
        "best" centers
        > Results from using this method, even with the
        same parameters, will vary slightly
        > Read the sklearn docs for more info
'''

# I've done testing, and have fixed k = 100 (best performance, on average)


def k_means_clustering(k,data):
    print("doing kmeans clustering calculations; this is non-deterministic")
    # Initialize the model with 2 parameters -- number of clusters and random state.
    kmeans_model = KMeans(n_clusters=k)  # default random state
    kmeans_model.fit(data)
    labels = kmeans_model.labels_  # get the cluster assignments
    best_k = k
    print("best k is " + str(best_k))
    return best_k, kmeans_model
