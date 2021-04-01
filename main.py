from structure import Camera, Scene_Set, Material, Light, Sphere, Plane, Box, Scene
import sys
import time


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
    # print(f'in_file: {in_file}\nout_file: {out_file}\nw: {width} , h: {hight}')
    return in_file, out_file, dimensions


def parse_scene(input_file_name, dimensions):
    materials = {}
    spheres = []
    boxes = []
    planes = []
    lights = []

    file = open(input_file_name, 'r')
    lines = [line.split() for line in file.readlines() if len(line.replace("\n", "")) > 0 and line[0] != '#']
    i, j, k, = 1, 1, 1

    for line in lines:
        # print(line)
        if line[0] == 'cam':
            # print('parsing camera settings...')
            camera = Camera(line[1:])
            fisheye, fishfactor = camera.fish_eye, camera.k
        elif line[0] == 'set':
            # print('parsing set...')
            scene_set = Scene_Set(line[1:])
        elif line[0] == 'mtl':
            # print(f'parsing material #{j}')
            materials[j] = Material(j, line[1:])
            j += 1
        elif line[0] == 'lgt':
            # print(f'parsing object #{i}')
            lights.append(Light(line[1:]))
            i += 1
        elif line[0] == 'sph':
            # print(f'parsing object #{k}')
            spheres.append(Sphere(line[1:], materials))
            k += 1
        elif line[0] == 'pln':
            # print(f'parsing object #{k}')
            planes.append((Plane(line[1:], materials)))
            k += 1
        elif line[0] == 'box':
            # print(f'parsing object #{k}')
            boxes.append(Box(line[1:], materials))
            k += 1

    shapes = {'spheres': spheres, 'planes': planes, 'boxes': boxes}
    print(f'Parsing Scene \'{input_file_name}\':\n\t> {k} Objects\n\t> {j} Materials\n\t> {i} Lights')
    fish_description = f'\t> Fish Eye effect enabled with k={k}' if fisheye else '\t> Fish Eye effect disabled'
    print(fish_description)

    scene = Scene(scene_set, camera, shapes, lights, dimensions, input_file_name)
    return scene


def main():
    hit_file = open('hits.txt', 'w')
    clk = time.time()
    input_file_name, out_name, dimensions = parse_args(sys.argv[1:])
    print(f'Screen size {dimensions[0]}x{dimensions[1]}')
    scene = parse_scene(input_file_name, dimensions)
    clk = time.time() - clk
    print(scene.screen.pixel_hits, file=hit_file)
    print(f'\nRay-Tracer.py() has finished running after {clk:.2f}s')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
