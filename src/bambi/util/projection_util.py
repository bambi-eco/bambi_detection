import json
import os.path
from enum import Enum
from typing import Optional

import cv2
import numpy as np
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

def get_sorted_mask_contour_pixels(mask_image):
    mask_image = mask_image[:,:, 0]
    mask_image = mask_image.astype(np.uint8)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    if len(contours) != 1:
        raise ValueError(f'Found {len(contours)} contours in mask image')

    return sort_contour_clockwise(contours[0])

def sort_contour_clockwise(contour):
    # Calculate centroid
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return contour
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Sort points by angle
    def angle_from_centroid(point):
        x, y = point[0]
        return np.arctan2(y - cy, x - cx)

    sorted_contour = sorted(contour, key=angle_from_centroid)
    return np.squeeze(np.array(sorted_contour), axis=1)

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

def combine_polygons(polygons, chunk_size: int = 4):
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

    return polygons[0]

def measure_area(mesh, intrinsic_matrix, extrinsics, mask, transformer, x_offset, y_offset, z_offset, chunk_size:int = 4, export_individual_polygons: bool = True, target_path_for_individual_polygons: Optional[str] = None):
    found_valid_polygons = False
    grouped_polygons = {}
    points3D = []
    contour_pixels = get_sorted_mask_contour_pixels(mask)

    for i, extrinsic in extrinsics:
        origins, directions = generate_rays(intrinsic_matrix, extrinsic, contour_pixels)
        directions *= -1
        # Raycast using trimesh
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False
        )

        sorted_indices = sorted(range(len(index_ray)), key=lambda k: index_ray[k])

        # Reorder both lists
        sorted_locations = [locations[i] for i in sorted_indices]
        points3D.extend(sorted_locations)

        if len(sorted_locations) >= 3:
            coordinates = []
            for p in sorted_locations:
                coordinates.append((p[0] + x_offset, p[1] + y_offset))
            try:
                if grouped_polygons.get(i) is None:
                    grouped_polygons[i] = []
                polygon = Polygon(coordinates)
                if polygon.is_valid:
                    found_valid_polygons = True
                    grouped_polygons[i].append(polygon)
                else:
                    print(f"Invalid polygon at frame {i}")
            except Exception as e:
                print(f"Failed to create polygon at frame {i}: {e}")

    if not found_valid_polygons:
        print("No valid polygons created")
        return {'area': 0, 'perimeter': 0}

    if export_individual_polygons and target_path_for_individual_polygons is not None:
        for group_id, polygon_group in grouped_polygons.items():
            combined_polygon = combine_polygons(polygon_group, chunk_size)
            local_gps_coordiantes = []
            xx = []
            yy = []
            zz = []
            for p in list(combined_polygon.exterior.coords):
                xx.append(p[0])
                yy.append(p[1])
                zz.append(0)
            transformed = transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
            for t in [x for x in zip(transformed[0], transformed[1])]:
                local_gps_coordiantes.append((t[1], t[0]))
            target_path = os.path.join(target_path_for_individual_polygons, f"{group_id}_area.geojson")
            print(f"Exporting area for frame {group_id} for {target_path}")
            with open(target_path, "w") as f:
                json.dump({
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "area": combined_polygon.area,
                                "perimeter": combined_polygon.length,
                                "fill": "#ffffff",
                                "fill-opacity": 0.5,
                                "stroke": "#ffffff",
                                "stroke-width": 2,
                                "stroke-opacity": 1,
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    local_gps_coordiantes
                                ]
                            }
                        }
                    ]
                }, f)

    # Union polygons
    print(f"Calculating overall area")
    polygons = [item for sublist in grouped_polygons.values() for item in sublist]
    unioned_polygon = combine_polygons(polygons, chunk_size)
    if isinstance(unioned_polygon, MultiPolygon):
        final_coordinates = list(unioned_polygon.geoms[0].exterior.coords)
    else:
        final_coordinates = list(unioned_polygon.exterior.coords)

    gps_coordiantes = []
    xx = []
    yy = []
    zz = []
    for p in final_coordinates:
        xx.append(p[0])
        yy.append(p[1])
        zz.append(0)
    transformed = transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
    for t in [x for x in zip(transformed[0], transformed[1])]:
        gps_coordiantes.append((t[1], t[0]))

    area = unioned_polygon.area
    perimeter = unioned_polygon.length

    return area, perimeter, gps_coordiantes


class ProjectionType(Enum):
    NoProjection = 0 # use if no projection should be applied for detection
    OrthographicProjection = 1 # use if only orthographic projection should be applied
    AlfsProjection = 2 # use if light field rendering should be applied