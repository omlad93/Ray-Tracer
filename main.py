import numpy as np
import structure
from structure import Camera, Set, Material, Light, Sphere, Plane, Box, Scene
import sys
import linalg


def parse_args(args):
    assert (2 <= len(args) <= 4)
    in_file = args[0]
    out_file = args[1]
    if len(args) == 2:
        width, hight = 500, 500
    elif len(args) == 3:
        width = args[2]
        hight = 500
    else:
        width = args[2]
        hight = args[3]
    # print(f'in_file: {in_file}\nout_file: {out_file}\nw: {width} , h: {hight}')
    return in_file, out_file, width, hight


def parse_scene(input_file_name):
    materials = {}
    spheres = []
    boxes = []
    planes = []
    lights = []

    file = open(input_file_name, 'r')
    lines = [line.split() for line in file.readlines() if len(line.replace("\n", "")) > 0 and line[0] != '#']
    j = 1
    for line in lines:
        # print(line)
        if line[0] == 'cam':
            print('parsing camera settings...')
            camera = Camera(line[1:])
        elif line[0] == 'set':
            print('parsing set...')
            scene_set = Set(line[1:])
        elif line[0] == 'mtl':
            print(f'parsing material #{j}')
            materials[j] = Material(j, line[1:])
            j += 1
        elif line[0] == 'lgt':
            lights.append(Light(line[1:]))
        elif line[0] == 'sph':
            spheres.append(Sphere(line[1:], materials))
        elif line[0] == 'pln':
            planes.append((Plane(line[1:], materials)))
        elif line[0] == 'box':
            boxes.append(Box(line[1:], materials))
    shapes = {'spheres': spheres, 'planes': planes, 'boxes': boxes}


def main():
    linalg.normalize(np.array([float('inf'), 2, 3, float('inf'), 5]))

    input_file_name, out_name, width, hight = parse_args(sys.argv[1:])
    parse_scene(input_file_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
