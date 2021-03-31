import numpy as np


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm != 0 and norm != float('inf'):
        return np.array([x / norm for x in vec])
    elif norm == 0:
        return vec
    elif norm == float('inf'):
        max = np.where(vec == np.amax(vec))[0]
        return [1 if j in max else 0 for j in range(len(vec))]
