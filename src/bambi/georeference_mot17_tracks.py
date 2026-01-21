"""
Geo-reference tracked bounding boxes in MOT format.

Input format (MOT):
    <frame-idx>, <track-id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

Output format:
    <frame-idx> <track-id> <utm-x-min> <utm-y-min> <utm-z-min> <utm-x-max> <utm-y-max> <utm-z-max> <conf> <class> [<visibility>]
"""

import json
import os
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
from alfspy.core.rendering import Resolution, Camera
from alfspy.core.util.geo import get_aabb
from alfspy.render.render import make_mgl_context, read_gltf, process_render_data, release_all
from pyproj.enums import TransformDirection
from pyrr import Quaternion, Vector3
from trimesh import Trimesh
from scipy.signal import savgol_filter

from src.bambi.util.projection_util import label_to_world_coordinates

from pyproj import CRS, Transformer


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


def get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation,
                         overrule_fov: Optional[float] = None):
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


def parse_mot_line(line: str) -> Optional[Tuple[int, int, float, float, float, float, float, int, float]]:
    """
    Parse a MOT format line.

    Input format: <frame-idx>, <track-id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

    Returns: (frame_idx, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility)
             or None if parsing fails
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 8:
        return None

    try:
        frame_idx = int(parts[0])
        track_id = int(parts[1])
        bb_left = float(parts[2])
        bb_top = float(parts[3])
        bb_width = float(parts[4])
        bb_height = float(parts[5])
        conf = float(parts[6])
        class_id = int(parts[7])
        visibility = float(parts[8]) if len(parts) > 8 else 1.0

        return frame_idx, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
    except (ValueError, IndexError):
        return None


def mot_bbox_to_corners(bb_left: float, bb_top: float, bb_width: float, bb_height: float) -> Tuple[float, float, float, float]:
    """
    Convert MOT bbox format (left, top, width, height) to corner format (x1, y1, x2, y2).

    Returns: (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right
    """
    x1 = bb_left
    y1 = bb_top
    x2 = bb_left + bb_width
    y2 = bb_top + bb_height
    return x1, y1, x2, y2


def extract_flight_id(filename: str) -> Optional[str]:
    """
    Extract flight ID from filename like '0_gt.txt' -> '0'.

    Args:
        filename: The filename (not full path)

    Returns:
        Flight ID string or None if pattern doesn't match
    """
    if not filename.endswith(".txt"):
        return None

    # Remove .txt extension
    name = filename[:-4]

    # Extract the number before the first underscore
    if "_" in name:
        flight_id = name.split("_")[0]
        # Verify it's a valid number
        if flight_id.isdigit():
            return flight_id

    return None


def georeference_mot_tracks(
    base_dir: str,
    correction_folder: str,
    target_base: str,
    additional_corrections_path: str,
    input_resolution: Resolution,
    skip_existing: bool = False,
    rel_transformer: Optional[Transformer] = None,
    transform_to_target_crs: bool = False,
    add_offsets: bool = True,
    apply_smoothing: bool = True,
    window_length: int = 11,
    polyorder: int = 2,
    include_visibility: bool = True,
    subfolders: List[str] = None
):
    """
    Main function to georeference MOT format tracking results.

    Expects folder structure:
        base_dir/
            test/
                0_gt.txt, 1_gt.txt, ...
            val/
                0_gt.txt, 1_gt.txt, ...
            train/
                0_gt.txt, 1_gt.txt, ...

    Args:
        base_dir: Root directory containing subfolders (test/val/train) with MOT files
        correction_folder: Directory containing correction data (poses, DEMs, etc.)
        target_base: Output directory for georeferenced results (same structure)
        additional_corrections_path: Path to JSON file with additional corrections
        input_resolution: Resolution of input frames
        skip_existing: Skip already processed files
        rel_transformer: Optional coordinate transformer
        transform_to_target_crs: Whether to transform to target CRS
        add_offsets: Whether to add DEM origin offsets
        apply_smoothing: Whether to apply pose smoothing
        window_length: Window length for smoothing
        polyorder: Polynomial order for smoothing
        include_visibility: Whether to include visibility in output
        subfolders: List of subfolders to process (default: ['test', 'val', 'train'])
    """

    if subfolders is None:
        subfolders = ['test', 'val', 'train']

    with open(additional_corrections_path) as f:
        all_additional_corrections = json.load(f)

    # Collect all files: list of (flight_id, input_path, output_path, subfolder)
    files_to_process = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Warning: Subfolder not found: {subfolder_path}")
            continue

        for filename in os.listdir(subfolder_path):
            flight_id = extract_flight_id(filename)
            if flight_id is None:
                continue

            input_path = os.path.join(subfolder_path, filename)
            output_folder = os.path.join(target_base, subfolder)
            output_path = os.path.join(output_folder, filename)

            files_to_process.append((flight_id, input_path, output_path, subfolder))

    # Group by flight_id for efficient processing (load correction data once per flight)
    flight_files = defaultdict(list)
    for flight_id, input_path, output_path, subfolder in files_to_process:
        flight_files[flight_id].append((input_path, output_path, subfolder))

    number_of_flights = len(flight_files)
    transformation_errors = 0
    total_tracks = 0

    for idx, (flight_id, file_list) in enumerate(flight_files.items()):
        print(f"Processing flight {flight_id}: {idx + 1} / {number_of_flights}")

        # Pre-check for remaining files
        remaining_files = []

        for input_path, output_path, subfolder in file_list:
            if skip_existing and os.path.exists(output_path):
                with open(input_path, "r", encoding="utf-8") as source, \
                     open(output_path, "r", encoding="utf-8") as target:
                    if len(source.readlines()) != len(target.readlines()):
                        remaining_files.append((input_path, output_path, subfolder))
                    else:
                        print(f"--- [{subfolder}] {os.path.basename(input_path)} already processed")
            else:
                remaining_files.append((input_path, output_path, subfolder))

        if len(remaining_files) == 0:
            print("--- all files already processed")
            continue

        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None

        try:
            # Load DEM metadata
            dem_meta_path = os.path.join(correction_folder, f"{flight_id}_dem_mesh_r2.json")
            if not os.path.exists(dem_meta_path):
                print(f"--- Warning: DEM metadata not found: {dem_meta_path}")
                continue

            with open(dem_meta_path, "r") as f:
                dem_meta = json.load(f)

            x_offset = dem_meta["origin"][0]
            y_offset = dem_meta["origin"][1]
            z_offset = dem_meta["origin"][2]

            # Load additional corrections if available
            additional_corrections = []
            if str(flight_id) in all_additional_corrections.get("corrections", {}):
                additional_corrections = all_additional_corrections["corrections"][str(flight_id)]

            # Initialize ModernGL context
            ctx = make_mgl_context()

            # Load poses
            poses_path = os.path.join(correction_folder, f"{flight_id}_matched_poses.json")
            if not os.path.exists(poses_path):
                print(f"--- Warning: Poses file not found: {poses_path}")
                continue

            with open(poses_path, "r") as f:
                poses = json.load(f)

            if apply_smoothing:
                smooth_pose_positions_savgol(poses, window_length=window_length, polyorder=polyorder)

            # Load correction data
            correction_path = os.path.join(correction_folder, f"{flight_id}_correction.json")
            if not os.path.exists(correction_path):
                print(f"--- Warning: Correction file not found: {correction_path}")
                continue

            with open(correction_path, 'r') as file:
                correction = json.load(file)

            translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
            rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})

            # Load digital elevation model
            path_to_dem = os.path.join(correction_folder, f"{flight_id}_dem.glb")
            if not os.path.exists(path_to_dem):
                print(f"--- Warning: DEM file not found: {path_to_dem}")
                continue

            mesh_data, texture_data = read_gltf(path_to_dem)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            mesh_aabb = get_aabb(mesh_data.vertices)

            # Process each MOT file for this flight
            for fidx, (input_path, output_path, subfolder) in enumerate(remaining_files):
                print(f"--- [{subfolder}] {fidx + 1}/{len(remaining_files)}: {os.path.basename(input_path)}")

                # Ensure output folder exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(input_path, "r", encoding="utf-8") as source, \
                     open(output_path, "w", encoding="utf-8") as target:

                    # Write header
                    if include_visibility:
                        target.write("# frame_idx track_id utm_x_min utm_y_min utm_z_min utm_x_max utm_y_max utm_z_max conf class visibility\n")
                    else:
                        target.write("# frame_idx track_id utm_x_min utm_y_min utm_z_min utm_x_max utm_y_max utm_z_max conf class\n")

                    for line in source:
                        parsed = parse_mot_line(line)
                        if parsed is None:
                            continue

                        frame_idx, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility = parsed
                        total_tracks += 1

                        # Convert MOT bbox to corner format
                        x1, y1, x2, y2 = mot_bbox_to_corners(bb_left, bb_top, bb_width, bb_height)

                        # Get frame-specific corrections
                        frame_translation = translation
                        frame_rotation = rotation
                        for additional_correction in additional_corrections:
                            if additional_correction["start frame"] < frame_idx < additional_correction["end frame"]:
                                frame_translation = additional_correction.get('translation', translation)
                                frame_rotation = additional_correction.get('rotation', rotation)
                                break

                        # Validate frame index
                        if frame_idx >= len(poses["images"]):
                            print(f"----- Warning: Frame {frame_idx} exceeds pose data length")
                            if include_visibility:
                                target.write(f"{frame_idx} {track_id} -1 -1 -1 -1 -1 -1 {conf} {class_id} {visibility}\n")
                            else:
                                target.write(f"{frame_idx} {track_id} -1 -1 -1 -1 -1 -1 {conf} {class_id}\n")
                            transformation_errors += 1
                            continue

                        # Get camera for frame
                        cor_rotation_eulers = Vector3([frame_rotation['x'], frame_rotation['y'], frame_rotation['z']], dtype='f4')
                        cor_translation = Vector3([frame_translation['x'], frame_translation['y'], frame_translation['z']], dtype='f4')
                        camera = get_camera_for_frame(poses, frame_idx, cor_rotation_eulers, cor_translation)

                        # Project bounding box corners to world coordinates
                        # Using 4 corner points: top-left, top-right, bottom-right, bottom-left
                        bbox_corners = [x1, y1, x2, y1, x2, y2, x1, y2]
                        world_coordinates = label_to_world_coordinates(
                            bbox_corners,
                            input_resolution,
                            tri_mesh,
                            camera
                        )

                        if len(world_coordinates) == 0:
                            if include_visibility:
                                target.write(f"{frame_idx} {track_id} -1 -1 -1 -1 -1 -1 {conf} {class_id} {visibility}\n")
                            else:
                                target.write(f"{frame_idx} {track_id} -1 -1 -1 -1 -1 -1 {conf} {class_id}\n")
                            transformation_errors += 1
                            continue

                        xx = world_coordinates[:, 0]
                        yy = world_coordinates[:, 1]
                        zz = world_coordinates[:, 2]

                        if transform_to_target_crs and rel_transformer is not None:
                            transformed = rel_transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
                            xx = np.array(transformed[0])
                            yy = np.array(transformed[1])
                            zz = np.array(transformed[2])

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

                            if include_visibility:
                                target.write(f"{frame_idx} {track_id} {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} {conf} {class_id} {visibility}\n")
                            else:
                                target.write(f"{frame_idx} {track_id} {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} {conf} {class_id}\n")
                        else:
                            if include_visibility:
                                target.write(f"{frame_idx} {track_id} -1 -1 -1 -1 -1 -1 {conf} {class_id} {visibility}\n")
                            else:
                                target.write(f"{frame_idx} {track_id} -1 -1 -1 -1 -1 -1 {conf} {class_id}\n")
                            transformation_errors += 1

        finally:
            # Free up resources
            if ctx is not None:
                release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh

    print(f"\n=== Summary ===")
    print(f"Total track entries processed: {total_tracks}")
    print(f"Transformation errors: {transformation_errors}")
    print(f"Success rate: {100 * (total_tracks - transformation_errors) / max(1, total_tracks):.2f}%")


if __name__ == '__main__':
    # Configuration
    # Expected structure:
    #   base_dir/
    #       test/
    #           0_gt.txt, 1_gt.txt, ...
    #       val/
    #           0_gt.txt, 1_gt.txt, ...
    #       train/
    #           0_gt.txt, 1_gt.txt, ...

    base_dir = r"Z:\Hugo\mot"
    correction_folder = r"Z:\correction_data"
    target_base = r"Z:\Hugo\mot_georeferenced"
    additional_corrections_path = r"Z:\correction_data\corrections.json"

    input_resolution = Resolution(1024, 1024)
    skip_existing = False

    # CRS transformation (optional)
    rel_transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32633))
    transform_to_target_crs = False
    add_offsets = True

    # Smoothing parameters
    apply_smoothing = True
    window_length = 11
    polyorder = 2

    # Output options
    include_visibility = True  # Set to False to omit visibility from output

    # Subfolders to process (set to None to use default ['test', 'val', 'train'])
    subfolders = ['test', 'val', 'train']

    # Run georeferencing
    georeference_mot_tracks(
        base_dir=base_dir,
        correction_folder=correction_folder,
        target_base=target_base,
        additional_corrections_path=additional_corrections_path,
        input_resolution=input_resolution,
        skip_existing=skip_existing,
        rel_transformer=rel_transformer,
        transform_to_target_crs=transform_to_target_crs,
        add_offsets=add_offsets,
        apply_smoothing=apply_smoothing,
        window_length=window_length,
        polyorder=polyorder,
        include_visibility=include_visibility,
        subfolders=subfolders
    )