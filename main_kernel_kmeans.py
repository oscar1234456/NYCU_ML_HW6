import numpy as np
from dataloader.dataloader import image_loader
from kernel.kernel import defined_kernel

# config
gamma_s = 1e-3
gamma_c = 1e-3

# load images data
pic, height, weight = image_loader("image1.png")  # pic:(10000, 3)

# design kernel (with announcement)
# using kernel get gram matrix
gram = defined_kernel(pic, gamma_s, gamma_c)

# kmeans function (diff types initialization)
# using kmeans function to get the clustering number
# plot the heatmaps

print()
