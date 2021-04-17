import numpy as np
import math
import structure
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
def first_intersection(lo, ray, shapes_dict, ignore_shape=None):
    first = INF
    closest = None
    hit_point = None
    for _, shape_list in shapes_dict.items():
        for shape in shape_list:
            point, t, intersect = shape.intersect(lo, ray)
            if shape != ignore_shape and (
                    intersect == Intersection.TANGENT or intersect == Intersection.THROUGH) and t < first:
                first = t
                hit_point = point
                closest = shape

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


# calculates the reflection vector R
def calc_r(vector, normal):
    r = (2 * np.dot(vector, normal) * normal) - vector
    return r


def single_light_diffuse_intensity(k, i, light_direction, normal):
    return i * k * max(np.dot(light_direction, normal), 0)


def single_light_specular_intensity(light, i, light_reflection, light_direction, n):
    return light.color * i * light.specular_intensity * (max(np.dot(light_reflection, light_direction), 0) ** n)


# calculates the diffuse intensity from all the lights in the scene
def get_diffuse_intensity(shape, point, light, i, background):
    if shape == None:
        return background
    intensity = np.array([0, 0, 0])
    light_direction = normalize(light.position - point)
    intensity = single_light_diffuse_intensity(light.color, i, light_direction, shape.get_normal(point))
    return intensity


# calculates the specular intensity  from all the lights in the scene
def get_specular_intensity(shape, point, light, i, camera_ray, background):
    if shape == None:
        return background
    intensity = np.array([0, 0, 0])
    light_direction = normalize(light.position - point)
    light_reflection = calc_r(light_direction, shape.get_normal(point))
    intensity = single_light_specular_intensity(light, i, light_reflection, light_direction, shape.material.phong_coeff)
    return intensity


# calculates the number of rays that are shooted from the grid that hit the surface
def calc_number_of_hits(light, grid, point, shape, shapes_dict):
    num_of_hits = 0
    for i in range(grid.num_of_cells):
        for j in range(grid.num_of_cells):
            current_cell_left_point = grid.bottom_left_point + i * grid.du + j * grid.dv
            current_ray_source_position = current_cell_left_point + np.array([np.random.uniform(0,grid.cell_size),np.random.uniform(0,grid.cell_size),0])
            #random_step_size = np.random.uniform(0, grid.cell_size, 2)  # two random semples
            #current_ray_source_position = current_cell_left_point + random_step_size  # check dimensions
            current_cell_ray = normalize(point - current_ray_source_position)
            (hit_point, first, closest) = first_intersection(current_ray_source_position, current_cell_ray, shapes_dict)
            if closest == shape:
                num_of_hits += 1
    return num_of_hits


def find_perpendicular_plane(light, current_light_ray):
    d = np.dot(current_light_ray, light.position)
    x = current_light_ray[0]
    y = current_light_ray[1]
    z = current_light_ray[2]
    u = np.array([1, 0, (-x + d) / z])
    v = np.array([0, 1, (-y + d) / z])
    plane = np.array([normalize(u), normalize(v)])
    return plane


def calc_shape_transparency(point, camera_ray, shape, shapes_dict, shape_color, background, lights, scene):
    if shape == None:
        return shape_color
    if shape.material.transparency == 0:
        return shape_color # the shape in not transparent
    else:
        (hit_point, first, closest) = first_intersection(point, camera_ray, shapes_dict, shape)
        shape_color = shape_color * (1 - shape.material.transparency) + get_stupid_color(hit_point, camera_ray, closest, shapes_dict, background, lights, scene) * shape.material.transparency
        while closest != None:
            if closest.material.transparency == 0:
                return shape_color
            else:
                transparency = closest.material.transparency
                (hit_point, first, closest) = first_intersection(hit_point, camera_ray, shapes_dict, closest)
                shape_color = shape_color * (1 - transparency) + get_stupid_color(hit_point, camera_ray, closest, shapes_dict,
                                                                                                 background, lights, scene) * transparency
        return shape_color


def calc_shape_reflection(point, ray, shape, shapes_dict, lights, scene, background, recursion):
    reflection_color = np.array([0,0,0])
    if shape == None or recursion == 0:
        return reflection_color
    else:
        # calculate the reflection ray R
        reflection_ray = ray - 2 * np.dot(ray, shape.get_normal(point)) * shape.get_normal(point)
        # calculate the intersection with the reflection ray
        (point, first, reflection_shape) = first_intersection(point, reflection_ray, shapes_dict, shape)
        color = get_stupid_color(point, ray, reflection_shape, shapes_dict, background, lights, scene)
        color = calc_shape_transparency(point, ray, reflection_shape, shapes_dict, color, background, lights, scene)
        return shape.material.reflection_color * (color + calc_shape_reflection(point, reflection_ray, reflection_shape, shapes_dict, lights, scene, background, recursion - 1))


# return a color to be shown on a pixel given:
#   point - a 3D point which is the first intersection of the ray
#   camera ray - the ray from camera for transperancy calculations
#   shape - the shape which the given point is located on
#   backround - the backround color to return if shape is None
#   recursion - the recursion level of reflection calculations
def get_color(intesection_result, scene, recursion=0):
    # if intesection_result is None:
    #     return np.array([0, 0, 0])
    shapes_dict = scene.shapes
    background = scene.general.background_color
    point, shape, camera_ray = intesection_result[0], intesection_result[2], intesection_result[3]
    color = get_stupid_color(point, camera_ray, shape, shapes_dict, background, scene)

    return np.clip(calc_shape_transparency(point, camera_ray, shape, shapes_dict, color, background, scene.lights, scene) + \
           calc_shape_reflection(point, camera_ray, shape, shapes_dict, scene.lights, scene, background, recursion), 0, 1)


def get_stupid_color(point, camera_ray, shape, shapes_dict, background, scene):
    color = np.array([0, 0, 0])
    if shape != None:
        for light in scene.lights:
            i = get_light_intensity(light, point, shape, scene.general.shadow_n, shapes_dict)
            diffuse_intensity = get_diffuse_intensity(shape, point, light, i, background)
            specular_intensity = get_specular_intensity(shape, point, light, i, camera_ray, background)
            color = color + shape.material.diffuse_color * diffuse_intensity + shape.material.specular_color * specular_intensity
        return np.clip(color, 0, 1)
    else:
        return np.clip(background,0,1)


# calculates the light intensity of a certain point
def get_shadow_intensity(light_list, point, camera_ray, shape, n, shapes_dict):
    total_intensity = 0
    # for each light source in the scene
    for light in light_list:
        current_light_ray = normalize(point - light.position)
        plane = find_perpendicular_plane(light, current_light_ray)
        grid = structure.Grid(light, n, plane)
        number_of_hits = calc_number_of_hits(light, grid, point, shape, shapes_dict)
        light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (number_of_hits / (n * n))
        total_intensity += light_intensity
    return total_intensity


def get_light_intensity(light, point, shape, n, shapes_dict):
    current_light_ray = normalize(point - light.position)
    plane = find_perpendicular_plane(light, current_light_ray)
    grid = structure.Grid(light, n, plane)
    number_of_hits = calc_number_of_hits(light, grid, point, shape, shapes_dict)
    percentage = (number_of_hits / (n * n))
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * percentage
    return light_intensity


def calc_effective_radius(k, theta, screen_dist):
    if k > 0 and k <= 1:
        return math.tan(k * theta) * screen_dist / k
    elif k == 0:
        return screen_dist * theta
    else:
        return math.sin(k * theta) * screen_dist / k

def calc_theta(k,radius,screen_dist):
    if k == 0:
        return math.radians(radius/screen_dist)
    elif k<=1 and k>0:
        return math.radians((1/k)*math.atan(math.radians(k*radius/screen_dist)))
    else:
        return  math.radians((1/k)*math.asin(math.radians(k*radius/screen_dist)))