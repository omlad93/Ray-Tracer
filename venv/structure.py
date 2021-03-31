import numpy as np
import linalg


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
        look_at_point = np.array(list(map(float, cam_args[3:6])))
        self.look_at_vector = linalg.normalize(np.subtract(look_at_point, self.position))
        self.up_vector = np.array(list(map(float, cam_args[6:9])))
        self.screen_dist = float(cam_args[9])
        self.screen_width = float(cam_args[10])
        if len(cam_args) == 11:
            self.fish_eye, self.k = False, 0.5
        elif len(cam_args) == 12:
            self.fish_eye, self.k = cam_args[11], 0.5
        elif len(cam_args) == 13:
            self.fish_eye, self.k = cam_args[11], float(cam_args[12])


class Screen:
    def __init__(self, cam_pos, up, look_at, dist, width=500, height=500):
        camera = cam_pos
        pixels = get_pixels(cam_pos, up, look_at, dist, width, height)


class Sphere:
    def __init__(self, sphr_args, material_mapping):
        self.center = np.array(list(map(float, sphr_args[0:3])))
        self.radius = float(sphr_args[3])
        self.material = material_mapping[int(sphr_args[4])]


class Box:
    def __init__(self, box_args, material_mapping):
        self.center = np.array(list(map(float, box_args[0:3])))
        self.scale = np.array(list(map(float, box_args[3:6])))
        self.rotation = np.array(list(map(float, box_args[6:9])))
        self.material = material_mapping[(box_args[9])]


class Plane:
    def __init__(self, pln_args, material_mapping):
        self.normal = np.array(list(map(float, pln_args[0:3])))
        self.offset = float(pln_args[3])
        self.material = material_mapping[int(pln_args[4])]


class Set:
    def __init__(self, set_args):
        self.backround_color = np.array(set_args[0:3])  # RGB
        self.shadow_n = set_args[3]
        self.recursion_depth = set_args[4]


class Scene:
    general: Set
    camera: Camera
    shapes: dict
    lights: Light
    screen: Screen

    def __init__(self, camera, shapes, ):
        pass
