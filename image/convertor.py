import numpy as np
from array2gif import write_gif


def colorpic2gif(history_colorpic, path):
    for i in range(len(history_colorpic)):
        history_colorpic[i] = np.transpose(history_colorpic[i], axes=(1, 0, 2))
    write_gif(history_colorpic, path, fps=1)
