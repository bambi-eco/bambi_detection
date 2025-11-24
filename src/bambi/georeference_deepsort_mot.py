import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
from alfspy.core.convert import world_to_pixel_coord
from alfspy.core.convert.convert import world_to_pixel_coord2
from alfspy.core.rendering import Resolution, Camera
from alfspy.core.util.geo import get_aabb
from alfspy.render.render import make_mgl_context, read_gltf, process_render_data, release_all
from numpy._typing import ArrayLike, NDArray
from pyproj.enums import TransformDirection
from pyrr import Quaternion, Vector3
from trimesh import Trimesh
from scipy.signal import savgol_filter

from src.bambi.util.projection_util import label_to_world_coordinates

from pyproj import CRS, Transformer

def smooth_pose_positions(poses, window_size: int = 11):
    """
    In-place smoothing of poses['images'][i]['location'] using a simple
    moving average over time.

    window_size must be odd.
    """
    images = poses["images"]
    n = len(images)
    if n == 0 or window_size < 2:
        return

    if window_size % 2 == 0:
        window_size += 1  # ensure odd

    # N x 3 array of positions
    positions = np.array([img["location"] for img in images], dtype=float)

    # simple moving average with edge padding
    pad = window_size // 2
    kernel = np.ones(window_size, dtype=float) / window_size

    padded = np.pad(positions, ((pad, pad), (0, 0)), mode="edge")

    smoothed = np.empty_like(positions)
    for dim in range(3):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")

    # write back into poses in-place
    for img, loc in zip(images, smoothed):
        img["location"] = loc.tolist()

def smooth_pose_positions_savgol(poses, window_length: int = 11, polyorder: int = 2):
    """
    Smooth the drone GPS positions in-place using a Savitzky–Golay filter.
    window_length must be odd and > polyorder.

    This replaces poses['images'][i]['location'] with smoothed values.
    """

    images = poses["images"]
    n = len(images)
    if n == 0:
        return

    # Ensure valid parameters
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_length < 3:
        return
    if window_length % 2 == 0:
        window_length += 1  # must be odd

    # Extract Nx3 matrix of positions
    positions = np.array([img["location"] for img in images], dtype=float)

    # Apply SavGol smoothing along time axis
    smoothed = savgol_filter(
        positions,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp"
    )

    # Write back to poses in-place
    for img, loc in zip(images, smoothed):
        img["location"] = loc.tolist()

def get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation, overrule_fov: Optional[float] = None):
    cur_frame_data = matched_poses['images'][frame_idx]
    fovy = cur_frame_data['fovy'][0]

    if overrule_fov is not None:
        fovy = overrule_fov

    position = Vector3(cur_frame_data['location'])
    rotation_eulers = (Vector3(
        [np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) - cor_rotation_eulers) * -1

    position += cor_translation
    rotation = Quaternion.from_eulers(rotation_eulers)

    return Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)

def deviating_folders(parent_path: str, sub_path: str) -> str:
    # If sub_path is a file, work with its directory
    folder = sub_path if os.path.isdir(sub_path) else os.path.dirname(sub_path)
    rel = os.path.relpath(folder, start=parent_path)
    return "" if rel == "." else rel

if __name__ == '__main__':
    # Paths
    base_dir = r"Z:\dets\source"
    correction_folder = r"Z:\correction_data"
    target_base = r"Z:\dets\georeferenced5"
    additional_corrections_path = r"Z:\correction_data\corrections.json"
    input_resolution = Resolution(1024, 1024)
    skip_existing = False
    rel_transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32633))
    transform_to_target_crs = False
    add_offsets = True

    # choose an odd window length, e.g. 11 frames, and low poly order
    apply_smoothing = True
    window_length = 11  # tune this based on your frame rate / flight dynamics
    polyorder = 2

    ##################################################################################

    with open(additional_corrections_path) as f:
        all_additional_corrections = json.load(f)

    # Dictionary parent_id -> list of files
    parent_dict = defaultdict(list)

    # Loop through both directories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt") and "_" in file:
                parent_id = file.split("_")[0]  # Extract parent ID before '_'
                full_path = os.path.join(root, file)
                parent_dict[parent_id].append(full_path)

    # Convert defaultdict to normal dict (optional)
    parent_dict = dict(parent_dict)

    number_of_flights = len(parent_dict)
    transformation_errors = 0
    total_bb = 0
    for idx, (parent, files) in enumerate(parent_dict.items()):
        print(f"Processing flight {parent}: {idx + 1} / {number_of_flights}")
        # precheck
        remaining_files = []
        number_of_files = len(files)
        for fidx, f in enumerate(files):
            # create target folder and files
            p = Path(f)
            target_folder = os.path.join(target_base, deviating_folders(base_dir, f))
            target_file = os.path.join(target_folder, p.name)
            if skip_existing and os.path.exists(target_file):
                with (open(f, "r", encoding="utf-8") as source,
                      open(os.path.join(target_folder, p.name), "r", encoding="utf-8") as target):
                    if len(source.readlines()) != len(target.readlines()):
                        remaining_files.append(f)
                    else:
                        print(f"--- {fidx + 1}/{number_of_files}: {f}")
                        print(f"----- {f} already processed")
            else:
                remaining_files.append(f)
        if len(remaining_files) == 0:
            print("--- all files already processed")
            continue
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
            # included = False
            # for fidx, f in enumerate(files):
            #     if f.endswith("26_3.txt"):
            #         included = True
            # if not included:
            #     continue
            with open(os.path.join(correction_folder, f"{parent}_dem_mesh_r2.json"), "r") as f:
                dem_meta = json.load(f)
            x_offset = dem_meta["origin"][0]
            y_offset = dem_meta["origin"][1]
            z_offset = dem_meta["origin"][2]

            additional_corrections = []
            if str(parent) in all_additional_corrections["corrections"]:
                additional_corrections = all_additional_corrections["corrections"][str(parent)]

            # initialize the ModernGL context
            ctx = make_mgl_context()
            with open(os.path.join(correction_folder, f"{parent}_matched_poses.json"), "r") as f:
                poses = json.load(f)

            if apply_smoothing:
                smooth_pose_positions_savgol(poses, window_length=window_length, polyorder=polyorder)

            with open(os.path.join(correction_folder, f"{parent}_correction.json"), 'r') as file:
                correction = json.load(file)
            translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
            rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})


            # load digital elevation model
            path_to_dem = os.path.join(correction_folder, f"{parent}_dem.glb")
            mesh_data, texture_data = read_gltf(path_to_dem)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            mesh_aabb = get_aabb(mesh_data.vertices)
            for fidx, f in enumerate(remaining_files):
                print(f"--- {fidx + 1}/{number_of_files}: {f}")
                # create target folder and files
                p = Path(f)
                target_folder = os.path.join(target_base, deviating_folders(base_dir, f))
                target_file = os.path.join(target_folder, p.name)
                os.makedirs(target_folder, exist_ok=True)
                os.makedirs(target_folder + "_reprojected", exist_ok=True) # todo remove
                with (open(f, "r", encoding="utf-8") as source,
                      open(os.path.join(target_folder, p.name), "w", encoding="utf-8") as target,
                      open(os.path.join(target_folder + "_reprojected", p.name), "w", encoding="utf-8") as target2): # todo remove
                    frame_labels = []
                    for idx, line in enumerate(source):
                        if idx == 0:
                            target.write(line)
                            continue
                        parts = line.split(" ")
                        frame = int(parts[0])
                        x1 = float(parts[1])
                        y1 = float(parts[2])
                        x2 = float(parts[3])
                        y2 = float(parts[4])
                        confidence = float(parts[5])
                        class_id = int(parts[6])
                        total_bb += 1

                        for additional_correction in additional_corrections:
                            if additional_correction["start frame"] < frame < additional_correction["end frame"]:
                                translation = additional_correction.get('translation', translation)
                                rotation = additional_correction.get('rotation', rotation)
                                break

                        image_metadata = poses["images"][frame]
                        fov = image_metadata["fovy"][0]

                        # project labels
                        cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
                        cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
                        camera = get_camera_for_frame(poses, frame, cor_rotation_eulers, cor_translation)
                        world_coordinates = label_to_world_coordinates([x1, y1, x2, y1, x2, y2, x1, y2],
                                                                       input_resolution, tri_mesh, camera)

                        if len(world_coordinates) == 0:
                            target.write(f"{idx} {frame} {-1} {-1} {-1} {-1} {-1} {-1} {confidence} {class_id}\n")
                            transformation_errors += 1
                            continue

                        xx = world_coordinates[:, 0]
                        yy = world_coordinates[:, 1]
                        zz = world_coordinates[:, 2]

                        if transform_to_target_crs:
                            xx2 = xx + x_offset
                            yy2 = yy + y_offset
                            zz2 = zz + z_offset
                            transformed = rel_transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
                            xx = transformed[0]
                            yy = transformed[1]
                            zz = transformed[2]

                        if len(xx) > 0 and len(yy) > 0 and len(zz) > 0:
                            if add_offsets:
                                xx = xx + x_offset
                                yy = yy + y_offset
                                zz = zz + z_offset
                            min_x = min(xx)
                            max_x = max(xx)
                            min_y = min(yy)
                            max_y = max(yy)
                            min_z = min(zz)
                            max_z = max(zz)
                            target.write(f"{idx} {frame} {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} {confidence} {class_id}\n")

                            # todo remove - start
                            pixel_coords = world_to_pixel_coord2(xx, yy, zz, input_resolution.width,
                                                                 input_resolution.height, camera)
                            xs = [p[0] if p is not None else -1 for p in pixel_coords]
                            ys = [p[1] if p is not None else -1 for p in pixel_coords]
                            try:
                                minx = min(xs)
                                maxx = max(xs)
                                miny = min(ys)
                                maxy = max(ys)
                                target2.write(
                                    f"{idx} {frame} {minx} {miny} {maxx} {maxy} {confidence} {class_id}\n")
                            except Exception:
                                target2.write(
                                    f"{idx} {frame} {-1} {-1} {-1} {-1} {confidence} {class_id}\n")
                            # todo remove - end
                        else:
                            target.write(f"{idx} {frame} {-1} {-1} {-1} {-1} {-1} {-1} {-1} {class_id}\n")
                            transformation_errors += 1
        finally:
            # free up resources
            release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
    print(f"Could not transform {transformation_errors} of {total_bb} bounding boxes")