from enum import Enum

import numpy as np
from alfspy.core.convert import pixel_to_world_coord
from alfspy.core.rendering import CtxShot
from alfspy.core.util.pyrrs import quaternion_from_eulers
from pyrr import Vector3


def create_shot(image, image_metadata, ctx, correction):
    position = Vector3(image_metadata["location"])
    rotation = image_metadata["rotation"]
    rotation = [val % 360.0 for val in rotation]
    rot_len = len(rotation)
    if rot_len == 3:
        eulers = [np.deg2rad(val) for val in rotation]
        rotation = quaternion_from_eulers(eulers, 'zyx')
    else:
        raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

    fov = image_metadata["fovy"][0]
    return CtxShot(ctx, image, position, rotation, fov, 1, correction, lazy=True)

def tile_image(img, tile_size):
    width, height, _ = img.shape

    # Calculate step size for overlapping tiles
    step_x = (width - tile_size) // 2
    step_y = (height - tile_size) // 2

    if step_x == 0 or step_y == 0:
        return [(0, 0, img)]

    # Calculate coordinates for the nine tiles
    tiles = []
    for y in [0, step_y, height - tile_size]:  # Top, middle, bottom rows
        for x in [0, step_x, width - tile_size]:  # Left, middle, right columns
            tiles.append((x, y))

    result = []
    # Create tiles and save them along with their labels
    for idx, (x, y) in enumerate(tiles):
        # Process image tile
        result.append((x, y, img[y:y + tile_size, x:x + tile_size]))
    return result

def label_to_world_coordinates(label_coordinates, input_resolution, tri_mesh, camera):
    pixel_xs = []
    pixel_ys = []
    for pixel_id, pixel in enumerate(label_coordinates):
        if pixel_id % 2 == 0:
            pixel_xs.append(int(float(pixel)))
        else:
            pixel_ys.append(int(float(pixel)))

    w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, input_resolution.width, input_resolution.height, tri_mesh, camera,
                                   include_misses=False)
    return w_poses

class ProjectionType(Enum):
    NoProjection = 0 # use if no projection should be applied for detection
    OrthographicProjection = 1 # use if only orthographic projection should be applied
    AlfsProjection = 2 # use if light field rendering should be applied