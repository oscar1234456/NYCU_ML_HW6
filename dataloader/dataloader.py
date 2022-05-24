import numpy as np
import cv2

FILEPATH = "./data/"


def image_loader(filename, filepath = FILEPATH):
    # convert pic(100*100*3) to numpy array (10000 * 3)
    # each pixel in the image should be treated as a data point
    pic = cv2.imread(filepath + filename) # pic: (H:100*W:100*Color:3)
    res = pic.reshape(10000, 3)
    print()


if __name__ == "__main__":
    image_loader("image1.png", "../data/")