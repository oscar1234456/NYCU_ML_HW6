import sys

import numpy as np

from image.show_image import show_colorpic
from kmeans.initialization import init_centroid
from scipy.spatial.distance import cdist

from visualization.visualization import make_visualization


def kmeans(gram, k, height=100, width=100, initial_mode=0):
    # gram:(10000 * 10000)
    # k: clusters number
    # initial_mode: the initialization methods

    print("____kmeans____")
    data_number = gram.shape[0]
    cluster_notation = np.zeros(data_number, dtype=np.uint8)  # cluster_notation:(10000,)

    centroids = init_centroid(gram, k, initial_mode)  # centroids:(k, feature_dim:gram.shape[1]=10000)
    eps = 1e-10
    diff = sys.maxsize
    max_iter = 10000
    count = 0
    history_color_pic = list()
    selected_color = np.random.choice(256, size=(k, 3))
    while count < max_iter and diff > eps:
        count += 1
        # E-step (Decide the cluster by now centroids)
        for index, now_points in enumerate(gram):
            distance = list()
            for now_focus_centroid in centroids:
                distance.append(cdist(now_points.reshape(1, -1), now_focus_centroid.reshape(1, -1), 'euclidean'))
            cluster_notation[index] = np.argmin(distance)

        # M-step
        new_centroids = np.zeros((k, gram.shape[1]))
        for now_focus_k in range(k):
            new_centroids[now_focus_k] = np.mean(gram[np.where(cluster_notation == now_focus_k)])
        diff = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids

        color_pic = make_visualization(selected_color, cluster_notation, height, width)
        history_color_pic.append(color_pic)

        print(f">>>Epoch {count}<<<")
        print(f"diff: {diff}")
        print(f">>>>>>>>><<<<<<<<<<")
        show_colorpic(color_pic, count)

    return cluster_notation, history_color_pic
