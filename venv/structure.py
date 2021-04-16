import numpy as np
import alg
from alg import Intersection
import time
import math
from itertools import count

INF = float('inf')


class Light:
    _idx = count(1)

    def __init__(self, light_args):
        self.name = 'Light(' + str(next(self._idx)) + ')'
        self.position = np.array(list(map(float, light_args[0:3])))
        self.color = np.array(list(map(float, light_args[3:6])))  # RGB
        self.specular_intensity = float(light_args[6])
        self.shadow_intensity = float(light_args[7])
        self.radius = float(light_args[8])

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Material:
    def __init__(self, index, mat_args):
        self.name = 'Material(' + str(index) + ')'
        self.index = index
        self.diffuse_color = np.array(list(map(float, mat_args[0:3])))
        self.specular_color = np.array(list(map(float, mat_args[3:6])))
        self.reflection_color = np.array(list(map(float, mat_args[6:9])))
        self.phong_coeff = float(mat_args[9])
        self.transparency = float(mat_args[10])

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Camera:
    def __init__(self, cam_args):
        self.position = np.array(list(map(float, cam_args[0:3])))
        self.towards_vector = alg.normalize(np.subtract(np.array(list(map(float, cam_args[3:6]))), self.position))
        self.up_vector = alg.normalize(np.array(list(map(float, cam_args[6:9]))))
        self.screen_dist = float(cam_args[9])
        self.screen_width = float(cam_args[10])
        self.right_vector = alg.normalize(np.cross(self.up_vector, self.towards_vector))
        if len(cam_args) == 11:
            self.fish_eye, self.k = False, 0.5
        elif len(cam_args) == 12:
            self.fish_eye, self.k = cam_args[11], 0.5
        elif len(cam_args) == 13:
            self.fish_eye, self.k = cam_args[11], float(cam_args[12])


class Sphere:
    _idx = count(1)
    intersection_time = 0
    intersections = 0

    def __init__(self, sphr_args, material_mapping):
        self.name = 'Sphere(' + str(next(self._idx)) + ')'
        self.center = np.array(list(map(float, sphr_args[0:3])))
        self.radius = float(sphr_args[3])
        self.material = material_mapping[int(sphr_args[4])]

    def intersect(self, lo, ray):
        """
        ray-Sphere intersection algorithem: ray is represented as lo + t*ray
        we assume there is a t such l0 + t*ray is intersecting with ray
        we get the formula for intersection and find roots
        if roots are positive we take minimum. if no postive root or no roots we declare miss
        The equation we recieve is the following equation:
        (ray^2)t^2 + 2(ray*(lo-center))*t + ((lo-center)*(lo-center) - offset)
        """
        int = time.time()
        Sphere.intersections += 1

        help_vec = np.subtract(lo, self.center)
        a = np.dot(ray, ray)
        b = 2 * np.dot(ray, help_vec)
        c = np.dot(help_vec, help_vec) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return (None, INF, Intersection.MISS)
        elif discriminant == 0:
            t = -b / (2 * a)
            point = lo + t * ray
            return (point, t, alg.Intersection.TANGENT)
        else:
            t = sorted([(math.sqrt(discriminant) - b) / (2 * a), -(b + math.sqrt(discriminant)) / (2 * a)])
            if (t[0] > 0):
                point = lo + t[0] * ray
                return (point, t[0], alg.Intersection.THROUGH)
            elif (t[1] > 0):
                point = lo + t[1] * ray
                return (point, t[1], alg.Intersection.THROUGH)
            return (None, INF, Intersection.MISS)
        Sphere.intersection_time += time.time() - Sphere.total_rooting

    def get_normal(self, point):
        normal = alg.normalize(np.subtract(point, self.center))
        return normal

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Box:
    _idx = count(1)

    def __init__(self, box_args, material_mapping):
        self.name = 'Box(' + str(next(self._idx)) + ')'
        self.center = np.array(list(map(float, box_args[0:3])))
        self.edge_length = float(box_args[3])
        self.material = material_mapping[int(box_args[4])]
        self.construct_planes()

    def intersect(self, lo, ray):
        first = INF
        closest_point = None
        for plane in planes:
            point, t, inetersection = plane.intersect(lo, ray)
            if inetersection == alg.Intersection.THROUGH:
                if self.bounds_point(point) and t < first:
                    first, closest_point, mode = t, point, inetersection
        return (first, t, mode)

    def bounds_point(self, point):
        max_dist = self.edge_length / 2
        xd = abs(point[0] - self.center[0])
        yd = abs(point[1] - self.center[1])
        zd = abs(point[2] - self.center[2])
        return xd < max_dist and yd < max_dist and zd < max_dist

    def construct_planes(self):
        self.planes = []
        step = 0.5 * self.edge_length
        plane_names = ['+x', '+y', '+z', '-x', '-y', '-z', ]
        unit_vectors = [np.array(1, 0, 0), np.array(0, 1, 0), np.array(0, 0, 1),
                        np.array(-1, 0, 0), np.array(0, -1, 0), np.array(0, 0, -1)]
        for i in range(6):
            normal = unit_vectors[i]
            point = self.center + step * normal
            self.planes.append(Plane(point, normal, plane_names[i], self))

    def get_normal(self, point):
        for plane in self.planes:
            if np.dot(point, plane.normal) == plane.offset:
                return plane.normal
        print('tried to get normal of a box for a non-edged point')
        exit(-2)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Plane:
    _idx = count(1)
    intersection_time = 0
    intersections = 0

    def __init__(self, pln_args=None, material_mapping=None,
                 from_box=False, point=None, normal=None, name=None, box=None):
        if not from_box:
            self.name = 'Plane(' + str(next(self._idx)) + ')'
            self.normal = alg.normalize(np.array(list(map(float, pln_args[0:3]))))
            self.offset = float(pln_args[3])
            self.material = material_mapping[int(pln_args[4])]
        if from_box:
            self.name = f'Plane.from_{box.name}({name})'
            self.normal = normal
            self.offset = np.dot(point, unit_vectors)
            self.material = box.material

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_normal(self, point):
        return self.normal

    def intersect(self, lo, ray):
        """
        ray-Plane intersection algorithem: ray is represented as lo + t*ray
        if ray is not parralell to the plane we set po a point on the plane
        we use po from the plane and p from the ray to find the t parameter for intersection
        t = (po-l0)*n/(ray*n)

        if ray is parralell to plane there are 2 options:
            Miss - lo not on Plane
            Unified - lo on Plane
        """
        Plane.intersections += 1
        int = time.time()
        dot_product = np.dot(self.normal, ray)
        if dot_product == 0:  # paralel or miss
            Plane.intersection_time += time.time() - int
            if np.dot(lo, self.normal) + self.offset == 0:
                return (lo, 0, Intersection.UNIFIED)
            else:
                return (None, INF, Intersection.MISS)
        else:
            po = self.offset * self.normal
            t = np.dot(np.subtract(po, lo), self.normal) / dot_product
            Plane.intersection_time += time.time() - int
            if t > 0:
                point = lo + t * ray
                return (point, t, Intersection.THROUGH)
            else:
                return (None, INF, Intersection.MISS)


class Scene_Set:
    def __init__(self, set_args):
        self.background_color = np.array(list(map(float, set_args[0:3])))  # RGB
        self.shadow_n = int(set_args[3])
        self.recursion_depth = int(set_args[4])


class Screen:
    def __init__(self, scene, dimensions, delta=1):
        self.name = scene.name.replace('Scene', 'Screen')
        self.Z = scene.camera.towards_vector  # Z Axis
        self.X = scene.camera.right_vector  # X Axis
        self.Y = -1 * scene.camera.up_vector  # Y Axis
        self.X_pixels = dimensions[0]
        self.Y_pixels = dimensions[1]
        self.pixel_size = scene.camera.screen_width / dimensions[0]
        self.width = scene.camera.screen_width
        self.hight = self.pixel_size * self.Y_pixels
        dx = self.pixel_size * self.X  # a pixel-sized step on X axis
        dy = self.pixel_size * self.Y  # a pixel-sized step on Y axis
        self.screen_center = scene.camera.position + scene.camera.screen_dist * scene.camera.towards_vector
        self.bottom_left_pixel_center = self.screen_center - 0.5 * (
                (self.width - self.pixel_size) * self.X + (self.hight - self.pixel_size) * self.Y)
        self.pixel_centers, self.pixel_rays = [], []
        self.pixel_hits, self.pixel_colors = [], []

        # if scene.camera.fish_eye:
        #     fish_sensor = [[np.zeros(3) for j in range(dimensions[1])] for i in range(dimensions[0])]
        #     bl_corner = self.bottom_left_pixel_center - 0.5 * self.pixel_size * self.X - 0.5 * self.pixel_size * self.Y

        total_intersection = 0
        total_ray_time = 0
        max_intersect_time = 0

        self.pixel_colors = [
                    [self.ret_color(self.px(i,j,scene), first_intersection(self.px(i,j,scene)[0], self.px(i,j,scene)[1], scene.shapes), scene.shapes, scene.general.background_color,scene.lights, scene) for j in range(dimensions[1])]
                        for i in range(dimensions[1])]

        # for i in range(dimensions[0]):  # for each row
        #     row_pixels = []
        #     row_colors = []
        #     # row_rays = []
        #     # row_hits = []
        #
        #     for j in range(dimensions[1]):  # for each column
        #
        #         current_pixel_center = self.bottom_left_pixel_center + j * dx + i * dy
        #         row_pixels.append(current_pixel_center)
        #
        #         ray_time = time.time()
        #         current_pixel_ray = alg.normalize(current_pixel_center - scene.camera.position)
        #         ray_time = time.time() - ray_time
        #         total_ray_time += ray_time
        #
        #         interssection_time = time.time()
        #         point, t, shape = first_intersection(current_pixel_center, current_pixel_ray, scene.shapes)
        #         interssection_time = time.time() - interssection_time
        #         total_intersection += interssection_time
        #         max_intersect_time = max(interssection_time, max_intersect_time)
        #
        #         current_pixel_color = shape.material.specular_color if shape is not None else scene.general.background_color
        #         # current_pixel_color = alg.get_stupid_color(point, current_pixel_ray, shape, scene.shapes,
        #         # scene.general.background_color, scene.lights,
        #         # scene)
        #         if scene.camera.fish_eye:
        #             ni, nj = self.fish_eye_transofrm(current_pixel_center, scene.camera.screen_dist, scene.camera.k,
        #                                              bl_corner, i, j)
        #             fish_sensor[ni][nj] = current_pixel_color
        #
        #         # row_colors.append(np.array(list(map(int,
        #         #                                     255 * current_pixel_color))))  # should be modified after Iris will give real color in return
        #         row_colors.append(np.array(current_pixel_color))
        #
        #     self.pixel_centers.append(row_pixels)
        #     self.pixel_colors.append(row_colors)
        #     # self.pixel_rays.append(row_rays)
        #     # self.pixel_hits.append(row_hits)
        if scene.camera.fish_eye:
            self.pixel_colors = fish_sensor
        # print(f'{self} is up and complete'
        #       f'\n\t* avrege ray creation time: {total_ray_time / (dimensions[0] * dimensions[1]):.4f}s, total:{total_ray_time:.4f}s'
        #       f'\n\t* avrege Sphere intersection time: {Sphere.intersection_time / Sphere.intersections:.4f}s, total:{Sphere.intersection_time:.4f}s'
        #       f'\n\t* avrege Plane  intersection time: {Plane.intersection_time / Plane.intersections:.4f}s, total:{Sphere.intersection_time:.4f}s'
        #       f'\n\t* avrege intersection time: {total_intersection / (dimensions[0] * dimensions[1]):.4f}s, total:{total_intersection:.4f}s, max:{max_intersect_time:.4f}s\n')

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Class.Screen.' + self.name

    def fish_eye_transofrm(self, originial_pix, screen_dist, fish_factor, bottom_left, i, j):
        p_x, p_y, _ = originial_pix
        c_x, c_y, c_z = self.screen_center
        dx, dy = abs(c_x - p_x), abs(c_y - p_y)
        r = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan(r / screen_dist)  # angle with Z axis
        R = alg.calc_effective_radius(fish_factor, theta, screen_dist)

        phi = math.atan(dy / dx)  # axis with X axis
        new_pix = (self.screen_center + R * math.cos(phi) * self.X + R * math.sin(phi) * self.Y)
        dist_from_bl = (new_pix - bottom_left) / self.pixel_size

        ni, nj = int(abs(dist_from_bl[0])), int(abs(dist_from_bl[1]))
        return ni, nj

    def fish_eye_transormation(self, i, j):
        pass

    def px(self, i, j, scene):
        center =  self.bottom_left_pixel_center + i * (self.pixel_size * self.X) + j * (self.pixel_size * self.Y)
        ray = alg.normalize(center - scene.camera.position)
        return center,ray

    def ret_color(self, px, first_intersection, shapes, background_color, lights,scene):
        shape = first_intersection[2]
        return shape.material.specular_color if shape is not None else background_color

class Grid:
    def __init__(self, light, n, plane):
        self.axis_U = plane[0]
        self.axis_V = plane[1]
        self.num_of_cells = n
        self.center = light.position
        self.width = light.radius
        self.cell_size = self.width / self.num_of_cells
        self.du = self.cell_size * self.axis_U
        self.dv = self.cell_size * self.axis_V
        self.bottom_left_point = np.array(self.center - 0.5 * (self.width * self.axis_U + self.width * self.axis_V))

    def __str__(self):
        return self.name


class Scene:

    def __init__(self, scene_set, camera, shapes_dict, light_list, dimensions, name):
        self.name = name.replace('txt', 'Scene')
        self.general = scene_set
        self.camera = camera
        self.shapes = shapes_dict
        self.lights = light_list
        self.screen = Screen(scene=self, dimensions=dimensions)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Class.Scene.' + self.name





def first_intersection(lo, ray, shapes_dict, ignore_shape=None, recursion=0):
    first = INF
    closest = None
    hit_point = None
    # intersections =[]
    for _, shape_list in shapes_dict.items():
        for shape in shape_list:
            point, t, intersect = shape.intersect(lo, ray)
            if shape != ignore_shape and (
                    intersect == Intersection.TANGENT or intersect == Intersection.THROUGH) and t < first:
                first = t
                hit_point = point
                closest = shape
            # if t>0 and t < INF:
            #     intersections.append((point,shape))

    return (hit_point, first, closest)  # , intersections)
