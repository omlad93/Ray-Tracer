from structure import Camera, Scene_Set, Material, Light, Sphere, Plane, Box, Scene
import sys
import numpy as np
import time
from PIL import Image


# parse command-line argument
def parse_args(args):
    assert (2 <= len(args) <= 4)
    in_file = args[0]
    out_file = args[1]
    if len(args) == 2:
        dimensions = (500, 500)
    elif len(args) == 3:
        dimensions = (int(args[2]), 500)
    else:
        dimensions = (int(args[2]), int(args[3]))
    return in_file, out_file, dimensions


# parse scene TXT file and generate a Scene instance
def parse_scene(input_file_name, dimensions):
    materials = {}
    spheres, boxes = [], []
    planes, lights = [], []
    camera, scene_set = None, None
    fisheye, fish_factor = None, 0
    try:
        file = open(input_file_name, 'r')
    except FileNotFoundError as err:
        print(f'\n\t Raised Error: {err}')
        exit(-2)
    lines = [line.split() for line in file.readlines() if len(line.strip().replace("\n", "")) > 0 and line[0] != '#']
    i, j, k = 1, 1, 1

    for line in lines:
        # print(line)
        if line[0] == 'cam':
            # print('parsing camera settings...')
            camera = Camera(line[1:])
            fisheye, fish_factor = camera.fish_eye, camera.k
        elif line[0] == 'set':
            # print('parsing set...')
            scene_set = Scene_Set(line[1:])
        elif line[0] == 'mtl':
            materials[j] = Material(j, line[1:])
            j += 1
        elif line[0] == 'lgt':
            lights.append(Light(line[1:]))
            i += 1
        elif line[0] == 'sph':
            spheres.append(Sphere(line[1:], materials))
            k += 1
        elif line[0] == 'pln':
            planes.append((Plane(line[1:], materials)))
            k += 1
        elif line[0] == 'box':
            boxes.append(Box(line[1:], materials))
            k += 1

    shapes = {'spheres': spheres, 'planes': planes, 'boxes': boxes}
    print(f'Parsing Scene from \'{input_file_name}\':\n\t> {k-1} Objects\n\t> {j-1} Materials\n\t> {i-1} Lights')
    fish_description = f'\t> Fish Eye effect enabled with k={fish_factor}' if fisheye else \
        '\t> Fish Eye effect disabled'
    print(fish_description)

    scene = Scene(scene_set, camera, shapes, lights, dimensions, input_file_name)
    return scene


# generate a PNG file according to Scene instance
def generate_png(png_path, scene):
    print(f'Generating a PNG file for {scene}')
    pix_array = np.array(scene.screen.pixel_colors)
    dim = (scene.screen.X_pixels, scene.screen.Y_pixels)
    pix_array.reshape(dim[0], dim[1], 3)
    png = Image.fromarray(np.uint8(pix_array))
    png.save(png_path)
    png.show()


def main():
    hit_file = open('hits.txt', 'w')
    clk = time.time()
    input_file_name, out_name, dimensions = parse_args(sys.argv[1:])
    print(f'Screen size {dimensions[0]}x{dimensions[1]}')
    scene = parse_scene(input_file_name, dimensions)
    process = time.time() - clk
    clk = time.time()
    generate_png(out_name, scene)
    saving = time.time() - clk
    print(f'Ray-Tracer.py has finished:\n\t- creating {out_name} took {process:.2f}s\n\t- saving took {saving:.2f}s')



if __name__ == '__main__':
    main()
