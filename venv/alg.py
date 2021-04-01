import numpy as np
INF = float('inf')

class Intersection(Enum):
    UNIFIED = 3
    THROUGH = 2
    TANGENT = 1
    MISS = 0

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm != 0 and norm != INF:
        return np.array([x / norm for x in vec])
    elif norm == 0:
        return vec
    elif norm == INF:
        max = np.where(vec == np.amax(vec))[0]
        return [1 if j in max else 0 for j in range(len(vec))]


def first_intersection(ray, shapes_dict):
    first = INF
    for shape_type, shape_list in shapes_dict.items():
        for shape in shape_list:

                print (f' (!) Unkown Shape Type: {shape_type}')
            pass

