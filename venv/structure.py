import numpy as np
import alg
from alg import Intersection
import time

Delta = 0.01
MAX_RAY = 100


class Light:
    def __init__(self, light_args):
        self.position = np.array(list(map(float, light_args[0:3])))
        self.color = np.array(list(map(float, light_args[3:6])))  # RGB
        self.specular_inetnsity = float(light_args[6])
        self.shadow_intesity = float(light_args[7])
        self.radius = float(light_args[8])


class Material:
    def __init__(self, index, mat_args):
        self.index = index
        self.diffuse_color = np.array(list(map(float, mat_args[0:3])))
        self.specular_color = np.array(list(map(float, mat_args[3:6])))
        self.reflection_color = np.array(list(map(float, mat_args[6:9])))
        self.phong_coeff = float(mat_args[9])
        self.transparency = float(mat_args[10])


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
    def __init__(self, sphr_args, material_mapping):
        self.center = np.array(list(map(float, sphr_args[0:3])))
        self.radius = float(sphr_args[3])
        self.material = material_mapping[int(sphr_args[4])]

    def intersect(self, lo, ray):
        t = Delta
        while t < MAX_RAY:
            point = lo + t * ray
            sum_squares = sum(np.power(np.subtract(self.center, point), 2))
            r2 = self.radius ** 2
            if (sum_squares < r2):
                return (point, Intersection.THROUGH)
            elif (sum_squares == r2):
                return (point, Intersection.TANGENT)
            t += Delta
        return (NONE, Intersection.MISS)


class Box:
    def __init__(self, box_args, material_mapping):
        self.center = np.array(list(map(float, box_args[0:3])))
        self.scale = np.array(list(map(float, box_args[3:6])))
        self.rotation = np.array(list(map(float, box_args[6:9])))
        self.material = material_mapping[(box_args[9])]

    def intersect(self, point):
        return (None,Intersection.MISS)


class Plane:
    def __init__(self, pln_args, material_mapping):
        self.normal = np.array(list(map(float, pln_args[0:3])))
        self.offset = float(pln_args[3])
        self.material = material_mapping[int(pln_args[4])]

    def intersect(self, lo, ray, t):
        dot_product = sum(np.multiply(self.normal, ray))
        if dot_product == 0: # paralel or miss
            if sum(np.multiply(lo,self.normal)) + self.offset == 0:
                return (lo, 0, Intersection.UNIFIED)
            else:
                return (None, alg.INF, Intersection.MISS)
        else:       # lets say po= (0,0,0)
            t = sum(np.multiply(-1*lo, self.normal))/dot_product
            point = lo + t * ray
            return (point, t, Intersection.THROUGH)




class Scene_Set:
    def __init__(self, set_args):
        self.backround_color = np.array(set_args[0:3])  # RGB
        self.shadow_n = set_args[3]
        self.recursion_depth = set_args[4]


# TODO
class Screen:
    def __init__(self, scene, dimensions, delta=1):
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
        for i in range(dimensions[0]):  # for each row
            current_pixels_row = []
            current_ray_row = []
            for j in range(dimensions[1]):  # for each column
                current_pixel_center = bottom_left_pixel_center + j * dx + i * dy
                current_pixel_ray = alg.normalize(current_pixel_center - scene.camera.position)
                current_pixels_row.append((current_pixel_center))
                current_ray_row.append(current_pixel_ray)
            self.pixel_centers.append(current_pixels_row)
            self.pixel_rays.append(current_ray_row)


class Scene:

    def __init__(self, scene_set, camera, shapes_dict, light_list, dimensions, ):
        self.general = scene_set
        self.camera = camera
        self.shapes = shapes_dict
        self.lights = light_list
        self.screen = Screen(scene=self, dimensions=dimensions)
        print(f'Scene is Ready for Compute')

    def calculate_hits(self, t=0.01):
        screen = self.screen
        for i in range(screen.X_pixels):
            for j in range(screen.Y_pixels):
                ray = screen.pixel_rays[i][j]
                hit = alg.first_intersection(ray, self.shapes)

        for shape_type, shape_list in self.shapes.items():
            print(f'Scene has {len(shape_list)} {shape_type}')
