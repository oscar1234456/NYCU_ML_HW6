import numpy as np
from scipy.spatial.distance import pdist, squareform


def defined_kernel(X, gamma_s, gamma_c):
    # kernel: 10000 * 10000
    data_number = X.shape[0]
    S = np.zeros((data_number, 2))  # S: the coordinate of the pixel
    for i in range(data_number):
        S[i, 0] = i // 100
        S[i, 1] = i % 100
    S_ = squareform(np.exp((-gamma_s) * pdist(S, 'sqeuclidean')))
    C_ = squareform(np.exp((-gamma_c) * pdist(X, 'sqeuclidean')))
    kernel = S_ * C_

    return kernel
