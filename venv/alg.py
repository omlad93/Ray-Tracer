import numpy as np
import math
from enum import Enum

INF = float('inf')


class Intersection(Enum):
    TANGENT = 3
    THROUGH = 2
    UNIFIED = 1
    MISS = 0


# normalizing vectors
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm != 0 and norm != INF:
        return np.array([x / norm for x in vec])
    elif norm == 0:
        return vec
    elif norm == INF:
        max = np.where(vec == np.amax(vec))[0]
        return [1 if j in max else 0 for j in range(len(vec))]


# finds first intersection of a ray with a shape stored in shaped_dict
# can be used recursivly for Transperacy & Reflection
# lo - the source of the ray, ray - normalized ray vector
def first_intersection(lo, ray, shapes_dict, recursion=0):
    first = INF
    closest = None
    hit_point = None
    for _, shape_list in shapes_dict.items():
        for shape in shape_list:
            point, t, intersect = shape.intersect(lo, ray)
            if (intersect == Intersection.TANGENT or intersect == Intersection.THROUGH) and t < first:
                first = t
                hit_point = point
                closest = shape
    if closest == None or type(closest) == type(shapes_dict['planes'][0]):
        pass
    # if closest != None:
    #     if closest.material.transparency != 0:
    #         first_intersection(point, ray,shapes_dict, recursion)
    #         pass
    return (hit_point, first, closest)


# finds a root of a 2nd degree equation given a,b coeiffecients and discriminant
# used for calculating intersection with spheres
def find_roots(a, b, discriminant):
    if discriminant < 0:
        return (Intersection.MISS, None)
    elif discriminant == 0:
        return (Intersection.TANGENT, -b / (2 * a))
    else:
        t = sorted([(math.sqrt(discriminant) - b) / (2 * a), -(b + math.sqrt(discriminant)) / (2 * a)])
        return (Intersection.THROUGH, t[0], t[1])


# TODO:
# return a color to be shown on a pixel given:
#   point - a 3D point which is the first intersection of the ray
#   camera ray - the ray from camera for transperancy calculations
#   shape - the shape which the given point is located on
#   backround - the backround color to return if shape is None
def get_color(point, camera_ray, shape, backround):
    if (shape != None):
        return shape.material.diffuse_color
    else:
        return backround
