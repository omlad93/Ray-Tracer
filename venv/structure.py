import numpy as np
import alg
from alg import Intersection
import time
from itertools import count

Delta = 0.1
MAX_RAY = 10


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
        help_vec = np.subtract(lo, self.center)
        a = sum(np.multiply(ray, ray))
        b = 2 * sum(np.multiply(ray, help_vec))
        c = sum(np.multiply(help_vec, help_vec)) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        result = alg.find_roots(a, b, discriminant)
        if result[0] == alg.Intersection.THROUGH and result[1] > 0:
            point = lo + result[1] * ray
            return (point, result[1], alg.Intersection.THROUGH)
        elif result[0] == alg.Intersection.THROUGH and result[2] > 0:
            point = lo + result[2] * ray
            return (point, result[2], alg.Intersection.THROUGH)
        elif result[0] == alg.Intersection.TANGENT and result[1] > 0:
            point = lo + result[1] * ray
            return (point, result[1], alg.Intersection.TANGENT)
        else:
            return (None, alg.INF, alg.Intersection.MISS)

    def get_normal(self, point):
        normal = np.subtract(point, self.center)
        normal = alg.normalize(normal)
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

    def intersect(self, point):
        return (None, alg.INF, Intersection.MISS)

    def get_normal(self, point):
        return self.center  # FIXME

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Plane:
    _idx = count(1)

    def __init__(self, pln_args, material_mapping):
        self.name = 'Plane(' + str(next(self._idx)) + ')'
        self.normal = alg.normalize(np.array(list(map(float, pln_args[0:3]))))
        self.offset = float(pln_args[3])
        self.material = material_mapping[int(pln_args[4])]

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
        dot_product = sum(np.multiply(self.normal, ray))
        if dot_product == 0:  # paralel or miss
            if sum(np.multiply(lo, self.normal)) + self.offset == 0:
                return (lo, 0, Intersection.UNIFIED)
            else:
                return (None, alg.INF, Intersection.MISS)
        else:
            po = self.offset * self.normal
            t = sum(np.multiply(np.subtract(po, lo), self.normal)) / dot_product
            if t > 0:
                point = lo + t * ray
                return (point, t, Intersection.THROUGH)
            else:
                return (None, alg.INF, Intersection.MISS)


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
        self.Y = scene.camera.up_vector  # Y Axis
        self.X_pixels = dimensions[0]
        self.Y_pixels = dimensions[1]
        self.pixel_size = scene.camera.screen_width / dimensions[0]
        self.width = scene.camera.screen_width
        self.hight = self.pixel_size * self.Y_pixels
        dx = self.pixel_size * self.X  # a pixel-sized step on X axis
        dy = self.pixel_size * self.Y  # a pixel-sized step on Y axis
        screen_center = scene.camera.position + scene.camera.screen_dist * scene.camera.towards_vector
        bottom_left_pixel_center = screen_center - 0.5 * (
                (self.width - self.pixel_size) * self.X + (self.hight - self.pixel_size) * self.Y)
        self.pixel_centers, self.pixel_rays = [], []
        self.pixel_hits, self.pixel_colors = [], []
        for i in range(dimensions[0]):  # for each row
            row_pixels = []
            row_rays = []
            row_hits = []
            row_colors = []

            for j in range(dimensions[1]):  # for each column
                current_pixel_center = bottom_left_pixel_center + j * dx + i * dy
                row_pixels.append(current_pixel_center)

                current_pixel_ray = alg.normalize(current_pixel_center - scene.camera.position)
                row_rays.append(current_pixel_ray)

                point, t, shape = alg.first_intersection(current_pixel_center, current_pixel_ray, scene.shapes)
                hit = (point, shape)
                row_hits.append(hit)

                current_pixel_color = alg.get_color(point, current_pixel_ray, shape, scene.shapes,
                                                    scene.general.background_color, scene.lights,
                                                    scene.general.recursion_depth)
                #current_pixel_color = current_pixel_color * alg.get_light_intensity(scene.lights, point,
                 #                                                                   current_pixel_ray, shape)
                row_colors.append(np.array(list(map(int,
                                                    255 * current_pixel_color))))  # should be modified after Iris will give real color in return

            self.pixel_centers.append(row_pixels)
            self.pixel_rays.append(row_rays)
            self.pixel_hits.append(row_hits)
            self.pixel_colors.append(row_colors)
        print(f'{self} is up and complete')

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Class.Screen.' + self.name


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
        self.bottom_left_point = np.array(self.center - 0.5*(self.width*self.self.axis_U + self.width*self.axis_V))

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
