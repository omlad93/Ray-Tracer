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
def first_intersection(lo, ray, shapes_dict, ignore_shape=None, recursion=0):
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


# calculates the reflection vector R
def calc_r(vector, normal):
    r = (2 * np.dot(vector, normal) * normal) - vector
    return r


def single_light_diffuse_intensity(k, i, light_direction, normal):
    return i * k * abs(np.dot(light_direction, normal))


def single_light_specular_intensity(light, i, light_reflection, camera_ray, n):
    return light.color * i * light.specular_intensity * abs(np.dot(light_reflection, camera_ray * (-1))) ** n


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
    intensity = single_light_specular_intensity(light, i, light_reflection, camera_ray, shape.material.phong_coeff)
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


# return a color to be shown on a pixel given:
#   point - a 3D point which is the first intersection of the ray
#   camera ray - the ray from camera for transperancy calculations
#   shape - the shape which the given point is located on
#   backround - the backround color to return if shape is None
#   recursion - the recursion level of reflection calculations
def get_color(point, camera_ray, shape, shapes_dict, background, lights, recursion):
    trans_background = background
    if (shape != None):
        # if transparency != 0 then calculate the background color (not nessesary the background color of the scene).
        if shape.material.transparency != 0:
            see_troughs = []
            (hit_point, first, closest) = first_intersection(point, camera_ray, shapes_dict, shape)
            see_troughs.append((hit_point, closest))
            while closest != None:
                if closest.material.transparency != 0:
                    (hit_point, first, closest) = first_intersection(hit_point, camera_ray, shapes_dict, closest)
                    # store the results (touples of shape and point) in array
                    see_troughs.append((hit_point, closest))
                else:
                    break
            if closest == None:
                trans_background = get_deep_background_color(see_troughs, point, camera_ray, background)
            else:
                trans_background = get_deep_background_color(see_troughs, point, camera_ray,
                                                             closest.material.diffuse_color)
        # calculate the reflection and get it's color
        # reflection_color = shape.material.reflection_color
        ray = camera_ray
        reflection_shape = shape
        reflections = []
        reflect_background = background
        while recursion != 0 and reflection_shape != None:
            # calculate the reflection ray R
            ray = calc_r(ray * (-1), reflection_shape.get_normal(point))
            # calculate the intersection with the reflection ray
            (point, first, reflection_shape) = first_intersection(point, ray, shapes_dict, reflection_shape)
            if reflection_shape == None:
                reflect_background = background
                break
            if recursion > 0:
                reflections.append((reflection_shape, point, ray))
            else:
                diffuse_color = reflection_shape.material.diffuse_color * get_diffuse_intensity(reflection_shape, point,
                                                                                                lights,
                                                                                                reflect_background)
                specular_color = reflection_shape.material.specular_color * get_specular_intensity(reflection_shape,
                                                                                                   point, lights, ray,
                                                                                                   reflect_background)
                # diffuse_color = reflection_shape.material.diffuse_color
                # specular_color = reflection_shape.material.specular_color
                reflect_background = (diffuse_color + specular_color) * (1 - shape.material.transparency)
            recursion -= 1
        reflection_color = get_reflection_color(reflections, reflect_background, lights)
        diffuse_color = shape.material.diffuse_color * get_diffuse_intensity(shape, point, lights, trans_background)
        # diffuse_color = shape.material.diffuse_color
        specular_color = shape.material.specular_color * get_specular_intensity(shape, point, lights, camera_ray,
                                                                                trans_background)
        # specular_color = shape.material.specular_color
        output_color = background * shape.material.transparency + \
                       (diffuse_color + specular_color) * (
                               1 - shape.material.transparency) + reflection_color * shape.material.reflection_color
        return output_color
    else:
        return background


def get_stupid_color(point, camera_ray, shape, shapes_dict, background, lights, scene):
    trans_background = np.array([0, 0, 0])
    reflect_background = np.array([0, 0, 0])
    diffuse_intensity = np.array([0, 0, 0])
    specular_intensity = np.array([0, 0, 0])
    if shape != None:
        for light in lights:
            i = get_light_intensity(light, point, shape, scene.general.shadow_n, shapes_dict)
            diffuse_intensity = diffuse_intensity + get_diffuse_intensity(shape, point, light, i, background)
            specular_intensity = specular_intensity + get_specular_intensity(shape, point, light, i, camera_ray, background)


        diffuse_color = shape.material.diffuse_color * diffuse_intensity
        specular_color = shape.material.specular_color * specular_intensity
        output_color = trans_background * shape.material.transparency + \
                       (diffuse_color + specular_color) * (1.0 - shape.material.transparency) +\
                       reflect_background * shape.material.reflection_color

        return output_color
    else:
        return background


def get_deep_background_color(deeper_list, point, camera_ray, background):
    for i in range(len(deeper_list)):
        current_shape = deeper_list[-(i + 1)]
        background = current_shape.material.transparency * background + \
                     (current_shape.material.diffuse_color + current_shape.material.specular_color) * (
                             1 - current_shape.material.material.transparency)
    return background


def get_reflection_color(reflection_list, background, lights):
    for i in range(len(reflection_list) - 1):
        current_shape = reflection_list[-(i + 1)][0]
        current_point = reflection_list[-(i + 1)][1]
        reflection_ray = reflection_list[-(i + 1)][2]

        diffuse_color = current_shape.material.diffuse_color * get_diffuse_intensity(current_shape, current_point,
                                                                                     lights, background)
        specular_color = current_shape.material.specular_color * get_specular_intensity(current_shape, current_point,
                                                                                        lights, reflection_ray,
                                                                                        background)
        specular_color = current_shape.material.specular_color
        background = current_shape.material.transparency * background + \
                     (diffuse_color + specular_color) * (1 - current_shape.material.transparency)
        background = reflection_list[-(i + 2)][0].material.reflection_color * background
    return background


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
