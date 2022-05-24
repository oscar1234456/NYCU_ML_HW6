import numpy as np


def make_visualization(selected_color, cluster_notation, height=100, width=100):
    # pick k colors
    color_pic = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            belong_k_for_pix = cluster_notation[h * width + w]
            color_pic[h, w, :] = selected_color[belong_k_for_pix]
    return color_pic.astype(dtype=np.uint8)
