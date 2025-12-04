import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyrr import Quaternion, Vector3
from scipy.signal import savgol_filter
from alfspy.core.rendering import Resolution, Camera
from alfspy.core.convert.convert import world_to_pixel_coord2


# --- Helpers ---

def smooth_pose_positions_savgol(poses, window_length: int = 11, polyorder: int = 2):
    images = poses["images"]
    n = len(images)
    if n == 0: return
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_length < 3: return
    if window_length % 2 == 0: window_length += 1

    positions = np.array([img["location"] for img in images], dtype=float)
    smoothed = savgol_filter(positions, window_length=window_length, polyorder=polyorder, axis=0, mode="interp")
    for img, loc in zip(images, smoothed):
        img["location"] = loc.tolist()


def get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation, overrule_fov=None):
    try:
        cur_frame_data = matched_poses['images'][frame_idx]
    except IndexError:
        return None

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
    folder = sub_path if os.path.isdir(sub_path) else os.path.dirname(sub_path)
    rel = os.path.relpath(folder, start=parent_path)
    return "" if rel == "." else rel


def get_box_corners(x1, y1, z1, x2, y2, z2):
    # 8 corners of the 3D box
    xs = [x1, x2, x1, x2, x1, x2, x1, x2]
    ys = [y1, y1, y2, y2, y1, y1, y2, y2]
    zs = [z1, z1, z1, z1, z2, z2, z2, z2]
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(zs, dtype=np.float32)


if __name__ == '__main__':
    # --- CONFIGURATION ---
    tracked_csv_dir = r"Z:\dets\georeferenced5_interpolated"
    correction_folder = r"Z:\correction_data"
    additional_corrections_path = r"Z:\correction_data\corrections.json"
    output_base_dir = r"Z:\dets\georeferenced5_interpolated_reprojected"

    # Ensure this matches your original footage (e.g., 4000x3000).
    # Using 1024 if the original was different might skew aspect ratio if not 1:1.
    input_resolution = Resolution(1024, 1024)

    apply_smoothing = True
    window_length = 11
    polyorder = 2
    # ---------------------

    print(f"Loading corrections from {additional_corrections_path}...")
    with open(additional_corrections_path) as f:
        all_additional_corrections = json.load(f)

    files_to_process = []
    for root, dirs, files in os.walk(tracked_csv_dir):
        for file in files:
            if file.endswith(".csv"):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No .csv files found in target directory.")
        sys.exit()

    for idx, full_file_path in enumerate(files_to_process):
        filename = os.path.basename(full_file_path)
        print(f"\n[{idx + 1}/{len(files_to_process)}] Processing: {filename}")

        if "_" in filename:
            parent_id = filename.split("_")[0]
        else:
            print(f"Skipping {filename}: Cannot determine parent ID (expected format 'PARENT_CHILD.csv')")
            continue

        # Load Metadata
        try:
            dem_path = os.path.join(correction_folder, f"{parent_id}_dem_mesh_r2.json")
            poses_path = os.path.join(correction_folder, f"{parent_id}_matched_poses.json")
            corr_path = os.path.join(correction_folder, f"{parent_id}_correction.json")

            with open(dem_path, "r") as f:
                dem_meta = json.load(f)
            x_offset = dem_meta["origin"][0]
            y_offset = dem_meta["origin"][1]
            z_offset = dem_meta["origin"][2]
            print(f"  > Loaded Offsets: X={x_offset:.2f}, Y={y_offset:.2f}, Z={z_offset:.2f}")

            with open(poses_path, "r") as f:
                poses = json.load(f)
            if apply_smoothing:
                smooth_pose_positions_savgol(poses, window_length=window_length, polyorder=polyorder)

            num_poses = len(poses['images'])
            print(f"  > Loaded Poses: {num_poses} frames available.")

            with open(corr_path, 'r') as f:
                base_correction = json.load(f)

            base_trans = base_correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
            base_rot = base_correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})

            # Time-dependent corrections
            flight_corrections = []
            if str(parent_id) in all_additional_corrections["corrections"]:
                flight_corrections = all_additional_corrections["corrections"][str(parent_id)]

        except FileNotFoundError as e:
            print(f"  ! Error loading metadata for {parent_id}: {e}")
            continue

        # Read CSV
        csv_data_by_frame = defaultdict(list)
        row_count = 0
        with open(full_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10: continue
                try:
                    frame = int(parts[0])
                    csv_data_by_frame[frame].append({
                        'tid': int(parts[1]),
                        'x1': float(parts[2]), 'y1': float(parts[3]), 'z1': float(parts[4]),
                        'x2': float(parts[5]), 'y2': float(parts[6]), 'z2': float(parts[7]),
                        'conf': float(parts[8]), 'cls': int(parts[9]),
                        'interp': int(parts[10]) if len(parts) > 10 else 0
                    })
                    row_count += 1
                except ValueError:
                    continue  # Skip header or malformed lines

        print(f"  > Loaded {row_count} rows from CSV spanning {len(csv_data_by_frame)} frames.")
        if row_count == 0:
            print("  ! CSV file appears empty or malformed.")
            continue

        # Debug Stats
        processed_objs = 0
        projected_objs = 0
        failed_pose = 0
        failed_proj = 0

        reprojected_lines = []
        sorted_frames = sorted(csv_data_by_frame.keys())

        # Process
        for frame in sorted_frames:
            # Check frame validity
            if frame >= num_poses:
                failed_pose += len(csv_data_by_frame[frame])
                # Print once if we run out of poses
                if failed_pose == len(csv_data_by_frame[frame]):
                    print(f"  ! Warning: CSV contains frame {frame}, but poses only has {num_poses} frames.")
                continue

            # Get Correction
            trans = base_trans
            rot = base_rot
            for ac in flight_corrections:
                if ac["start frame"] < frame < ac["end frame"]:
                    trans = ac.get('translation', trans)
                    rot = ac.get('rotation', rot)
                    break

            cor_rot = Vector3([rot['x'], rot['y'], rot['z']], dtype='f4')
            cor_trans = Vector3([trans['x'], trans['y'], trans['z']], dtype='f4')

            camera = get_camera_for_frame(poses, frame, cor_rot, cor_trans)
            if not camera:
                failed_pose += len(csv_data_by_frame[frame])
                continue

            for obj in csv_data_by_frame[frame]:
                processed_objs += 1

                # 1. Global to Local
                lx1, ly1, lz1 = obj['x1'] - x_offset, obj['y1'] - y_offset, obj['z1'] - z_offset
                lx2, ly2, lz2 = obj['x2'] - x_offset, obj['y2'] - y_offset, obj['z2'] - z_offset

                # 2. Get 8 corners
                c_xs, c_ys, c_zs = get_box_corners(lx1, ly1, lz1, lx2, ly2, lz2)

                # 3. Project (Explicit float32 casting for pyrr/gl interaction)
                pixel_coords = world_to_pixel_coord2(
                    c_xs, c_ys, c_zs,
                    input_resolution.width, input_resolution.height,
                    camera
                )

                # 4. Filter None values
                valid_px = [p for p in pixel_coords if p is not None]

                if not valid_px:
                    failed_proj += 1
                    # Debug first failure
                    if failed_proj == 1:
                        print(f"  ! First projection failure at frame {frame}. "
                              f"Local Coord Sample: ({lx1:.2f}, {ly1:.2f}, {lz1:.2f}). "
                              f"Check if this is within camera FOV.")
                    continue

                xs = [p[0] for p in valid_px]
                ys = [p[1] for p in valid_px]

                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Clamp
                min_x = max(0.0, min(min_x, input_resolution.width))
                max_x = max(0.0, min(max_x, input_resolution.width))
                min_y = max(0.0, min(min_y, input_resolution.height))
                max_y = max(0.0, min(max_y, input_resolution.height))

                # Check valid area
                if (max_x - min_x) > 1 and (max_y - min_y) > 1:
                    line = (f"{frame} {obj['tid']} {min_x:.2f} {min_y:.2f} {max_x:.2f} {max_y:.2f} "
                            f"{obj['conf']:.4f} {obj['cls']} {obj['interp']}")
                    reprojected_lines.append(line)
                    projected_objs += 1
                else:
                    # Object projected but was too small or off screen
                    failed_proj += 1

        print(f"  > Summary: Processed {processed_objs}, Projected {projected_objs}. "
              f"Failed Poses: {failed_pose}, Failed Projection/Clipped: {failed_proj}")

        # Save
        rel_path = deviating_folders(tracked_csv_dir, full_file_path)
        target_folder = os.path.join(output_base_dir, rel_path)
        os.makedirs(target_folder, exist_ok=True)

        out_name = Path(full_file_path).stem + ".txt"
        out_path = os.path.join(target_folder, out_name)

        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write("frame_id track_id x1 y1 x2 y2 conf cls interpolated\n")
            for line in reprojected_lines:
                f_out.write(line + "\n")

        if projected_objs > 0:
            print(f"  > Success! Output written to {out_path}")
        else:
            print(f"  ! Warning: Output file {out_path} created but contains 0 tracks.")

    print("\nBatch processing complete.")