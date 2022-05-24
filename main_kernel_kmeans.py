import os
import datetime
import numpy as np
from dataloader.dataloader import image_loader
from image.convertor import colorpic2gif
from kernel.kernel import defined_kernel

# config
from kmeans.kmeans import kmeans

gamma_s = 1e-3
gamma_c = 1e-3
k = 4
initial_mode = 1

pic_name = "image1.png"

gif_path = os.path.join('{}_{}Clusters_{}'.format(pic_name.split(".")[0], k, 'kernel k-means.gif'))

# load images data
pic, height, weight = image_loader(pic_name)  # pic:(10000, 3)

# design kernel (with announcement)
# using kernel get gram matrix
gram = defined_kernel(pic, gamma_s, gamma_c)

# kmeans function (diff types initialization)
notation, history_colorpic = kmeans(gram, k, initial_mode=initial_mode)
# using kmeans function to get the clustering number
# plot the heatmaps
now = datetime.datetime.now()
colorpic2gif(history_colorpic,
             "./lab/kernel_kmeans/" + str(now.month) + "-" + str(now.day) + "-" + str(now.hour) + "-" + str(now.minute) +"_init_"+str(initial_mode)+"_"+gif_path)
print()
