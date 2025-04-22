from collections import defaultdict
from enum import Enum

import cv2
import numpy as np
import trimesh
from alfspy.core.convert import pixel_to_world_coord
from alfspy.core.rendering import CtxShot
from alfspy.core.util.pyrrs import quaternion_from_eulers
from pyproj.enums import TransformDirection
from pyrr import Vector3
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import Polygon

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

def compute_visible_face_mask(mesh: trimesh.Trimesh,
                              mask: np.ndarray,
                              K: np.ndarray,
                              cam_ext: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array of shape (n_faces,), True if that face
    is both under 'mask' when viewed through (K, cam_ext) and visible.
    """
    F = len(mesh.faces)
    centroids = mesh.triangles_center          # (F,3)
    cam_center = cam_ext[:3, 3]
    R_wc = cam_ext[:3,:3]; R_cw = R_wc.T; t_cw = -R_cw @ cam_center

    # project centroids
    pts_cam = (R_cw @ centroids.T).T + t_cw
    uvw     = (K @ pts_cam.T).T
    uv      = uvw[:, :2] / uvw[:, 2:3]
    u_i     = np.round(uv[:,0]).astype(int)
    v_i     = np.round(uv[:,1]).astype(int)

    # mask check
    H, W = mask.shape
    inb = (u_i>=0)&(u_i<W)&(v_i>=0)&(v_i<H)
    white = np.zeros(F, bool)
    white[inb] = (mask[v_i[inb], u_i[inb]] > 0)
    candidates = np.nonzero(white)[0]
    if len(candidates) == 0:
        return np.zeros(F, bool)

    # occlusion test
    dirs = centroids[candidates] - cam_center
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    hits = mesh.ray.intersects_first(
        ray_origins    = np.tile(cam_center, (len(candidates),1)),
        ray_directions = dirs
    )
    visible = np.zeros(F, bool)
    # keep those where first‐hit == face‐index
    hit_mask = (hits == candidates)
    visible[candidates[hit_mask]] = True

    return visible


def combined_region_properties(mesh: trimesh.Trimesh,
                               mask: np.ndarray,
                               K: np.ndarray,
                               cam_exts: list):
    """
    masks, Ks, cam_exts must be same length.
    Returns:
      area      : float = summed surface area of unioned visible faces
      perimeter : float = total boundary length of that union
      loops_xyz : list of (Ni,3) arrays = 3D loops around the union
    """
    mask = mask[:,:,0]
    # 1) build combined face‐mask
    F = len(mesh.faces)
    combined_mask = np.zeros(F, bool)
    for ext in cam_exts:
        combined_mask |= compute_visible_face_mask(mesh, mask, K, ext)

    # assume `combined_mask` is your boolean array over mesh.faces
    face_indices = np.nonzero(combined_mask)[0]

    # 1) extract the submesh containing *only* those faces
    #    `append=True` will re‑index vertices so they're contiguous
    submesh = mesh.submesh([face_indices], append=True)

    # 2) true surface area
    area = submesh.area

    # 3) calculate perimeter
    path3d = submesh.outline()  # Path3D object

    # 4) each `path3d` may contain one or more discrete loops;
    #    `.discrete` returns a list of (N,3) arrays in order
    loops_xyz = path3d.discrete

    # 5) compute the perimeter of each loop, then pick the outermost
    perimeters = []
    for loop in loops_xyz:
        # ensure closed
        if not np.allclose(loop[0], loop[-1]):
            loop = np.vstack((loop, loop[0]))
        # sum segment lengths
        segs = loop[1:] - loop[:-1]
        perimeters.append(np.linalg.norm(segs, axis=1).sum())
    perimeter = sum(perimeters)
    return area, perimeter, submesh


def get_mask_contour_pixels(mask_image):
    mask_image = mask_image[:,:, 0]
    mask_image = mask_image.astype(np.uint8)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Flatten all contour points into a list of (x, y)
    contour_pixels = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            contour_pixels.append((x, y))
    return contour_pixels


def generate_rays(intrinsic, extrinsic, pixels):
    K_inv = np.linalg.inv(intrinsic)
    rays_origin = []
    rays_direction = []

    for (u, v) in pixels:
        pixel_hom = np.array([u, v, 1.0])
        direction_cam = K_inv @ pixel_hom  # Direction in camera space
        direction_cam /= np.linalg.norm(direction_cam)

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        direction_world = R @ direction_cam
        origin_world = t

        rays_origin.append(origin_world)
        rays_direction.append(direction_world / np.linalg.norm(direction_world))

    return np.array(rays_origin), np.array(rays_direction)

def measure_area(mesh, intrinsic_matrix, extrinsics, mask, transformer, x_offset, y_offset, z_offset, chunk_size=4):
    polygons = []
    points3D = []
    contour_pixels = get_mask_contour_pixels(mask)
    for i, extrinsic in enumerate(extrinsics):
        origins, directions = generate_rays(intrinsic_matrix, extrinsic, contour_pixels)
        directions *= -1
        # ray_lines = [trimesh.load_path([origin, origin + direction * 10]) for origin, direction in
        #              zip(origins, directions)]
        # scene = trimesh.Scene([mesh] + ray_lines)
        # scene.show()

        # Raycast using trimesh
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False
        )

        valid_points = locations.tolist()
        points3D.extend(valid_points)

        if len(valid_points) >= 3:
            coordinates = []
            xx = []
            yy = []
            zz = []
            for p in valid_points:
                xx.append(p[0] + x_offset)
                yy.append(p[1] + y_offset)
                zz.append(p[2] + z_offset)
                coordinates.append((p[0] + x_offset, p[1] + y_offset))
            # transformed = transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
            # for t in [x for x in zip(transformed[0], transformed[1])]:
            #     coordinates.append((t[1], t[0]))
            coordinates.append(coordinates[0])  # close polygon

            try:
                polygon = Polygon(coordinates)
                import matplotlib.pyplot as plt

                x, y = polygon.exterior.xy
                plt.plot(x, y)
                plt.show()
                polygons.append(polygon)
                # if polygon.is_valid:
                #     polygons.append(polygon)
                # else:
                #     print(f"Invalid polygon at frame {i}")
            except Exception as e:
                print(f"Failed to create polygon at frame {i}: {e}")

    if not polygons:
        print("No valid polygons created")
        return {'area': 0, 'perimeter': 0}

    # Union polygons
    while len(polygons) > 1:
        next_chunk_polygons = []
        for i in range(0, len(polygons), chunk_size):
            chunk = polygons[i:i + chunk_size]
            if len(chunk) == 1:
                next_chunk_polygons.append(chunk[0])
            else:
                unioned = unary_union(chunk)
                next_chunk_polygons.append(unioned)
        polygons = next_chunk_polygons

    unioned_polygon = polygons[0]

    if isinstance(unioned_polygon, MultiPolygon):
        final_coordinates = list(unioned_polygon.geoms[0].exterior.coords)
        print("Area Measurement: MultiPolygon detected, using first polygon for outline only!")
    else:
        final_coordinates = list(unioned_polygon.exterior.coords)

    area = unioned_polygon.area
    perimeter = unioned_polygon.length


    return {'area': area, 'perimeter': perimeter}


class ProjectionType(Enum):
    NoProjection = 0 # use if no projection should be applied for detection
    OrthographicProjection = 1 # use if only orthographic projection should be applied
    AlfsProjection = 2 # use if light field rendering should be applied