import sys

import numpy as np
from scipy.spatial.distance import cdist


def init_centroid(gram, k, initial_mode):
    data_number = gram.shape[0]
    feature_dim = gram.shape[1]
    centroids = None

    if initial_mode == 0:
        # random mode -> choose random points to be the centroid
        centroids = np.zeros((k, feature_dim))
        mean_feature_dim = np.mean(gram, axis=0)
        std_feature_dim = np.std(gram, axis=0)

        for now_focus_feature_dim in range(feature_dim):
            centroids[:, now_focus_feature_dim] = np.random.normal(mean_feature_dim[now_focus_feature_dim],
                                                                   std_feature_dim[now_focus_feature_dim], size=k)
    elif initial_mode == 1:
        # kmeans++ -> picked k centroids that far away each other
        # ref: https://www.geeksforgeeks.org/ml-k-means-algorithm/
        centroids = list()
        centroids.append(gram[np.random.randint(data_number), :])

        for _ in range(k - 1):
            # pick left k-1 centroids
            dist = list()
            for focus_data in gram:
                d = sys.maxsize
                for focus_centroid in centroids:
                    temp = cdist(focus_data.reshape(1, -1), focus_centroid.reshape(1, -1), 'euclidean')
                    d = min(d, temp)
                dist.append(d)
            dist = np.array(dist)
            new_centroid = gram[np.argmax(dist), :]
            centroids.append(new_centroid)

        centroids = np.array(centroids)
    return centroids
