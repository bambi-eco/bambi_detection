import json
import os

import numpy as np
import pandas as pd
from alfspy.core.convert import pixel_to_world_coord
from alfspy.core.rendering import Resolution
from pyrr import Vector3
from trimesh import Trimesh

from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.orthografic_projection import get_camera_for_frame, get_frame_correction
from alfspy.render.render import read_gltf, process_render_data

def read_bounding_boxes(file_path):
    """
    Reads bounding box definitions from a file and returns a pandas DataFrame.

    Args:
        file_path (str): Path to the bounding box file.

    Returns:
        pd.DataFrame: DataFrame containing bounding box information.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the first line (comment)
    data_lines = lines[1:]

    # Prepare lists for each column
    ids = []
    x_start = []
    y_start = []
    x_end = []
    y_end = []
    confidences = []
    classes = []

    # Parse each line
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) != 7:
            continue  # Skip malformed lines
        ids.append(parts[0])
        x_start.append(float(parts[1]))
        y_start.append(float(parts[2]))
        x_end.append(float(parts[3]))
        y_end.append(float(parts[4]))
        confidences.append(float(parts[5]))
        classes.append(int(parts[6]))

    # Create a pandas DataFrame
    return pd.DataFrame({
        'Frame_ID': ids,
        'X_Start': x_start,
        'Y_Start': y_start,
        'X_End': x_end,
        'Y_End': y_end,
        'Confidence': confidences,
        'Class': classes
    })

def check_file_row_equality(p1, p2):
    with open(p1, 'r') as f1, open(p2, 'r') as f2:
        lines_file1 = sum(1 for _ in f1)
        lines_file2 = sum(1 for _ in f2)
        return lines_file1 == lines_file2

if __name__ == '__main__':
    """
    Help script allowing to convert bounding boxes (like https://doi.org/10.5281/zenodo.15532567) from image coordinates 
    to world coordinates as e.g. required for our tracking experiments.
    Based on initial detection results in the format of
    
    # /sequences/test/6_1/img1
    00000849 787.21 830.92 824.91 901.20 0.6292 0
    00000851 791.21 826.79 825.10 894.69 0.8406 0
    00000851 749.04 815.38 783.88 868.43 0.7610 0
    00000852 790.87 824.24 825.38 893.03 0.7750 0
    00000852 749.64 812.83 785.38 865.27 0.5648 0
    00000853 791.27 821.60 825.61 890.08 0.7783 0
    ...
    
    for every video sequence
    """
    # path to the parent folder containing the splits with the specific bounding boxes
    bounding_box_base_path = r"C:\Users\P41743\Desktop\tmp\sequences\dets"
    # path to the correction data (DEM file, corrections, poses file, ...)
    correction_data_path = r"C:\Users\P41743\Desktop\tmp\sequences\correction_data"
    # file with additional flight corrections
    additional_corrections_path = r"C:\Users\P41743\Desktop\tmp\sequences\corrections.json"
    # resolution of the underlying images
    input_resolution = Resolution(1024, 1024)
    # target path where projected labels should be written to
    target_path = r"C:\Users\P41743\Desktop\tmp\sequences\dets_proj"


    ################################
    # load the additional flight corrections
    with open(additional_corrections_path, 'r') as file:
        additional_corrections = json.load(file)

    # go through the dataset splits of the bounding boxes and start processing
    for split in os.listdir(bounding_box_base_path):
        print(f"Processing {split}...")
        # prepare target folder
        os.makedirs(os.path.join(target_path, split), exist_ok=True)
        # for every flight there might be multiple annotated sequences. Every sequence has got its own label file
        # to reduce the number of times a DEM has to be loaded, group those sequences based on the origin flight
        flights = {}

        # get through the current split and get all of labels of our video sequences
        split_path = os.path.join(bounding_box_base_path, split)
        files_in_split = os.listdir(split_path)
        num_of_files_in_split = len(files_in_split)
        for file_dix, file_name in enumerate(files_in_split):
            print(f"{file_dix} / {num_of_files_in_split} ({split}): {file_name}...")
            file_path = os.path.join(split_path, file_name)
            # check if the labels have not already been processed
            target_file_path = os.path.join(target_path, split, os.path.basename(file_path))
            if os.path.exists(target_file_path) and check_file_row_equality(file_path, target_file_path):
                print("--- already processed. Skip.")
                continue

            if os.path.isfile(file_path):
                # read the current labels
                df = read_bounding_boxes(file_path)
                file_name = os.path.basename(file_path)
                flight_id = int(file_name.split('_')[0])

                # add the current labels to the specific flight
                if flights.get(flight_id) is None:
                    flights[flight_id] = []
                flights[flight_id].append((file_path, df))

        # now it is time to process the labels per flight
        for flight_id, labels in flights.items():
            # load the flight associated files like the DEM, the correction file and the pose file
            # 1. load poses
            pose_file = os.path.join(correction_data_path, f"{flight_id}_matched_poses.json")
            with open(pose_file, 'r') as file:
                poses = json.load(file)

            # 2. load corrections and combine the default flight corrections with the sequence specific corrections
            specific_correction_file = os.path.join(correction_data_path, f"{flight_id}_correction.json")
            with open(specific_correction_file, 'r') as file:
                corrections_data = {
                    "default": json.load(file),
                    "corrections": additional_corrections["corrections"].get(str(flight_id), [])
                }

            # 3. load the DEM file
            dem_file = os.path.join(correction_data_path, f"{flight_id}_dem.glb")
            mesh_data, texture_data = read_gltf(dem_file)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            mesh_aabb = get_aabb(mesh_data.vertices)

            # now lets actually process the labels
            for (file_path, df) in labels:
                # check if the labels have not already been processed
                target_file_path = os.path.join(target_path, split, os.path.basename(file_path))
                if os.path.exists(target_file_path) and check_file_row_equality(file_path, target_file_path):
                    print("--- already processed. Skip projection.")
                    continue

                # if labels have not been processed lets start
                num_of_bb = len(df)
                if num_of_bb == 0:
                    print(f"---- No bounding boxes")

                # convert every individual bounding box
                to_write = []
                for bb_idx, bb in enumerate(df.itertuples(index=False)):
                    print(f"---- Projecting BB {bb_idx} / {num_of_bb}")
                    # get the frame specific corrections of the location and orientation
                    frame_id = int(bb.Frame_ID)
                    frame_correction = get_frame_correction(corrections_data, frame_id)
                    translation = frame_correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
                    cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
                    rotation = frame_correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
                    cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')

                    # prepare the camera object
                    camera = get_camera_for_frame(poses, frame_id, cor_rotation_eulers, cor_translation)

                    # now project the current bounding box
                    pixel_xs = [bb.X_Start, bb.X_End, bb.X_End, bb.X_Start]
                    pixel_ys = [bb.Y_Start, bb.Y_Start, bb.Y_End, bb.Y_End]
                    world_coordinates = pixel_to_world_coord(pixel_xs, pixel_ys, input_resolution.width,
                                                             input_resolution.height, tri_mesh, camera,
                                                             include_misses=False)

                    # Get the result and write the projected bounding box to the target file
                    X_Start = min(world_coordinates[:,0])
                    X_End = max(world_coordinates[:,0])
                    Y_Start = min(world_coordinates[:, 1])
                    Y_End = max(world_coordinates[:, 1])
                    to_write.append(f"{bb.Frame_ID} {X_Start} {Y_Start} {X_End} {Y_End} {bb.Confidence} {bb.Class}")

                with (open(target_file_path, 'w') as target_file,
                      open(file_path) as original_file):
                    header_line = original_file.readline()
                    target_file.write(header_line)
                    for bb_idx, w in enumerate(to_write):
                        target_file.write(w)
                        if bb_idx < num_of_bb - 1:
                            target_file.write("\n")

                if not check_file_row_equality(file_path, target_file_path):
                    raise ValueError(f"Error: Unequal number of lines for projection of {file_path}.")
