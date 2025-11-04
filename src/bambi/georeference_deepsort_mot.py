import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from alfspy.core.rendering import Resolution
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.orthografic_projection import get_camera_for_frame
from alfspy.render.render import make_mgl_context, read_gltf, process_render_data, release_all
from pyproj.enums import TransformDirection
from pyrr import Quaternion, Vector3
from trimesh import Trimesh
from alfspy.core.geo import Transform

from src.bambi.util.projection_util import label_to_world_coordinates

from pyproj import CRS, Transformer

def deviating_folders(parent_path: str, sub_path: str) -> str:
    # If sub_path is a file, work with its directory
    folder = sub_path if os.path.isdir(sub_path) else os.path.dirname(sub_path)
    rel = os.path.relpath(folder, start=parent_path)
    return "" if rel == "." else rel

if __name__ == '__main__':
    # Paths
    base_dir = r"Z:\dets\source"
    correction_folder = r"Z:\correction_data"
    target_base = r"Z:\dets\georeferenced_wgs84"
    additional_corrections_path = r"Z:\correction_data\corrections.json"
    input_resolution = Resolution(1024, 1024)
    skip_existing = True
    rel_transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32633))
    transform_to_target_crs = True

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
    # Print result
    for idx, (parent, files) in enumerate(parent_dict.items()):
        print(f"Processing flight {parent}: {idx + 1} / {number_of_flights}")
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
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
            number_of_files = len(files)
            for fidx, f in enumerate(files):
                print(f"--- {fidx + 1}/{number_of_files}: {f}")
                # create target folder and files
                p = Path(f)
                target_folder = os.path.join(target_base, deviating_folders(base_dir, f))
                target_file = os.path.join(target_folder, p.name)
                if skip_existing and os.path.exists(target_file):
                    with (open(f, "r", encoding="utf-8") as source,
                          open(os.path.join(target_folder, p.name), "r", encoding="utf-8") as target):
                        if len(source.readlines()) == len(target.readlines()):
                            print("----- already processed")
                            continue
                os.makedirs(target_folder, exist_ok=True)
                with (open(f, "r", encoding="utf-8") as source,
                      open(os.path.join(target_folder, p.name), "w", encoding="utf-8") as target):
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
                        xx = world_coordinates[:, 0] + x_offset
                        yy = world_coordinates[:, 1] + y_offset
                        zz = world_coordinates[:, 2] + z_offset

                        if transform_to_target_crs:
                            transformed = rel_transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
                            xx = transformed[0]
                            yy = transformed[1]
                            zz = transformed[2]
                        min_x = min(xx)
                        max_x = max(xx)
                        min_y = min(yy)
                        max_y = max(yy)
                        min_z = min(zz)
                        max_z = max(zz)
                        target.write(f"{frame} {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} {confidence} {class_id}\n")
        finally:
            # free up resources
            release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
