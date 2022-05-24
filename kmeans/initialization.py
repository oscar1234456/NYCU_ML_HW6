import numpy as np


def init_centroid(gram, k, initial_mode):
    # random mode -> choose random points to be the centroid
    feature_dim = gram.shape[1]
    centroids = np.zeros((k, feature_dim))

    if initial_mode == 0:
        # random mode
        mean_feature_dim = np.mean(gram, axis=0)
        std_feature_dim = np.std(gram, axis=0)

        for now_focus_feature_dim in range(feature_dim):
            centroids[:, now_focus_feature_dim] = np.random.normal(mean_feature_dim[now_focus_feature_dim], std_feature_dim[now_focus_feature_dim], size=k)
    return centroids