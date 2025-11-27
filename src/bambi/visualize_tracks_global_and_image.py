import colorsys
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

import src.bambi.georeferenced_tracking as gt
from src.bambi.georeference_deepsort_mot import deviating_folders  # :contentReference[oaicite:2]{index=2}

from collections import defaultdict
import math

from src.bambi.video.video_writer import FFMPEGWriter


def build_sequence_tracks_from_results(results):
    """
    results: list of (frame, tid, Detection) from gt.track_detections()

    Returns:
      sequence_tracks: dict[str frame_id] -> list[dict(...)]
      global_extent:   (min_x, max_x, min_y, max_y)
    """
    sequence_tracks = defaultdict(list)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    # IMPORTANT: we enumerate in the same order as gt.write_tracks_csv()
    # so row_idx stays consistent with how you used it before.
    for row_idx, (frame, tid, d) in enumerate(results):
        frame_str = f"{frame:08d}"  # same zero-padding as write_tracks_csv()

        source_id = d.source_id
        x1, y1, z1 = d.x1, d.y1, d.z1
        x2, y2, z2 = d.x2, d.y2, d.z2
        conf = d.conf
        cls = d.cls

        min_x = min(min_x, x1, x2)
        max_x = max(max_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)

        sequence_tracks[frame_str].append(
            {
                "row_idx": row_idx,
                "tid": tid,
                "conf": conf,
                "cls": cls,
                "gx1": x1,
                "gy1": y1,
                "gx2": x2,
                "gy2": y2,
                "source_id": source_id
            }
        )

    if min_x == float("inf"):
        return {}, None

    global_extent = (min_x, max_x, min_y, max_y)
    return sequence_tracks, global_extent

def load_tracks_live_from_georef_txt(
    georef_txt_path,
    iou_thresh=0.3,
    class_aware=True,
    max_age=-1,
    tracker_mode=gt.TrackerMode.HUNGARIAN,
    max_center_distance=0.2,
    min_confidence=0.0,
):
    """
    Directly run tracking on a geo-referenced detection file and build the
    same structures as load_tracks_with_global_coords().
    """
    # read geo-referenced detections (UTM boxes)
    frames = gt.read_detections(georef_txt_path, min_confidence=min_confidence)

    # run your tracker "live"
    results = gt.track_detections(
        frames,
        iou_thr=iou_thresh,
        class_aware=class_aware,
        max_age=max_age,
        tracker_mode=tracker_mode,
        max_center_distance=max_center_distance,
    )

    # convert to sequence_tracks + global_extent
    return build_sequence_tracks_from_results(results)

# ============================================================
# Color + drawing helpers
# ============================================================

def id_to_color(identifier, *, saturation=0.65, lightness=0.5):
    """Deterministically map any identifier to a distinct BGR color."""
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2**32  # [0,1)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    # Bottom
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    # Left
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    # Right
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


# ============================================================
# Track CSV (global coords) loader  :contentReference[oaicite:3]{index=3}
# ============================================================

def load_tracks_with_global_coords(tracks_csv):
    """
    Read CSV written by georeferenced_tracking.write_tracks_csv:
      frame(08d), tid, x1,y1,z1,x2,y2,z2,conf,cls

    Returns:
      sequence_tracks: dict[str frame_id] -> list[dict(...)]
      global_extent: (min_x, max_x, min_y, max_y)
    """
    sequence_tracks = defaultdict(list)

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    with open(tracks_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue

            frame = row[0]  # zero-padded "00000001"
            tid = int(row[1])

            x1 = float(row[2])
            y1 = float(row[3])
            # z1 = float(row[4])
            x2 = float(row[5])
            y2 = float(row[6])
            # z2 = float(row[7])
            conf = float(row[8])
            cls = int(row[9])

            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)

            sequence_tracks[frame].append(
                {
                    "row_idx": row_idx,
                    "tid": tid,
                    "conf": conf,
                    "cls": cls,
                    "gx1": x1,
                    "gy1": y1,
                    "gx2": x2,
                    "gy2": y2,
                }
            )

    if min_x == float("inf"):
        return {}, None

    global_extent = (min_x, max_x, min_y, max_y)
    return sequence_tracks, global_extent


# ============================================================
# MOT-style detections in image space
# ============================================================

def read_detections_image_space(det_path):
    """
    Reads MOT-like detection file:
      frame x1 y1 x2 y2 confidence class_id
    """
    dets = []
    with open(det_path, "r", encoding="utf-8") as source:
        for idx, line in enumerate(source):
            if idx == 0:
                continue  # skip header if present
            parts = line.split()
            if len(parts) < 7:
                continue
            frame = parts[0]
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            confidence = float(parts[5])
            class_id = int(parts[6])
            # todo orig_id
            dets.append((frame, x1, y1, x2, y2, confidence, class_id))
    return dets


# ============================================================
# Pose smoothing (same idea as in georeference_deepsort_mot) :contentReference[oaicite:4]{index=4}
# ============================================================

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

    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_length < 3:
        return
    if window_length % 2 == 0:
        window_length += 1

    positions = np.array([img["location"] for img in images], dtype=float)
    smoothed = savgol_filter(
        positions,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp"
    )
    for img, loc in zip(images, smoothed):
        img["location"] = loc.tolist()


# ============================================================
# Global canvas config + mapping + axes
# ============================================================

def make_global_canvas(global_extent, width=800, height=800, margin=60):
    min_x, max_x, min_y, max_y = global_extent
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    scale_x = (width - 2 * margin) / span_x
    scale_y = (height - 2 * margin) / span_y
    scale = min(scale_x, scale_y)

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "scale": scale,
        "margin": margin,
        "width": width,
        "height": height,
    }


def world_to_canvas(x, y, canvas_cfg):
    """Map world coords (x,y) to pixel coords on the global canvas."""
    min_x = canvas_cfg["min_x"]
    min_y = canvas_cfg["min_y"]
    scale = canvas_cfg["scale"]
    margin = canvas_cfg["margin"]
    height = canvas_cfg["height"]

    px = int(margin + (x - min_x) * scale)
    # flip y so that higher world-y is "up"
    py = int(height - (margin + (y - min_y) * scale))
    return px, py


def draw_axes_on_canvas(map_img, canvas_cfg, num_ticks=4):
    """Draw X/Y axes and numeric ticks in world coordinates."""
    min_x = canvas_cfg["min_x"]
    max_x = canvas_cfg["max_x"]
    min_y = canvas_cfg["min_y"]
    max_y = canvas_cfg["max_y"]

    axis_color = (200, 200, 200)
    text_color = (255, 255, 255)

    # X-axis at y = min_y
    x_axis_start = world_to_canvas(min_x, min_y, canvas_cfg)
    x_axis_end = world_to_canvas(max_x, min_y, canvas_cfg)
    cv2.line(map_img, x_axis_start, x_axis_end, axis_color, 1)

    # Y-axis at x = min_x
    y_axis_start = world_to_canvas(min_x, min_y, canvas_cfg)
    y_axis_end = world_to_canvas(min_x, max_y, canvas_cfg)
    cv2.line(map_img, y_axis_start, y_axis_end, axis_color, 1)

    # X ticks
    if num_ticks < 2:
        num_ticks = 2
    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        x_val = min_x + t * (max_x - min_x)
        px, py = world_to_canvas(x_val, min_y, canvas_cfg)
        cv2.line(map_img, (px, py), (px, py + 5), axis_color, 1)
        label = f"{int(round(x_val))}"
        cv2.putText(
            map_img,
            label,
            (px - 30, py + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Y ticks
    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        y_val = min_y + t * (max_y - min_y)
        px, py = world_to_canvas(min_x, y_val, canvas_cfg)
        cv2.line(map_img, (px - 5, py), (px, py), axis_color, 1)
        label = f"{int(round(y_val))}"
        cv2.putText(
            map_img,
            label,
            (px - 80, py + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Axis labels
    h, w = map_img.shape[:2]
    cv2.putText(
        map_img,
        "X (UTM / world)",
        (w // 2 - 60, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        map_img,
        "Y (UTM / world)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Drone position loading from matched_poses + corrections :contentReference[oaicite:5]{index=5}
# ============================================================

def load_drone_positions_for_sequence(
    parent_id: str,
    frame_ids_for_sequence,
    correction_folder: str,
    additional_corrections_for_parent,
    apply_pose_smoothing: bool = False,
    window_length: int = 11,
    polyorder: int = 2,
    x_offset: float = 0,
    y_offset: float = 0,
    z_offset: float = 0
):
    """
    Compute drone (camera) world positions for the frames in this sequence.

    Uses:
      - {parent}_matched_poses.json -> poses["images"][frame]["location"]
      - {parent}_correction.json -> translation offset
      - additional_corrections_for_parent -> frame-range specific translation overrides
    """
    matched_poses_path = os.path.join(correction_folder, f"{parent_id}_matched_poses.json")
    correction_path = os.path.join(correction_folder, f"{parent_id}_correction.json")

    if not os.path.exists(matched_poses_path):
        print(f"  [WARN] No matched_poses for parent {parent_id}")
        return {}

    with open(matched_poses_path, "r", encoding="utf-8") as f:
        poses = json.load(f)

    if apply_pose_smoothing:
        smooth_pose_positions_savgol(poses, window_length=window_length, polyorder=polyorder)

    # base translation
    base_translation = {"x": 0.0, "y": 0.0, "z": 0.0}
    if os.path.exists(correction_path):
        with open(correction_path, "r", encoding="utf-8") as f:
            corr = json.load(f)
        base_translation = corr.get("translation", base_translation)

    def effective_translation(frame_idx: int):
        t = base_translation
        for ac in additional_corrections_for_parent:
            if ac["start frame"] < frame_idx < ac["end frame"]:
                t = ac.get("translation", t)
        return t

    drone_positions = {}
    images = poses.get("images", [])

    for frame_id in frame_ids_for_sequence:
        try:
            frame_idx = int(frame_id)
        except ValueError:
            continue

        if not (0 <= frame_idx < len(images)):
            continue

        img_meta = images[frame_idx]
        loc = np.array(img_meta["location"], dtype=float)

        t = effective_translation(frame_idx)
        offset = np.array([t["x"] + x_offset, t["y"] + y_offset, t["z"] + z_offset], dtype=float)

        pos_world = loc + offset
        drone_positions[frame_id] = pos_world  # (x, y, z)

    return drone_positions


# ============================================================
# MAIN: live visualization over multiple sequences
# ============================================================

if __name__ == "__main__":
    georef_dets_base_folder = r"Z:\dets\georeferenced5" #r"Z:\dets\georeferenced_smoothed"
    detections_base_folder = r"Z:\dets\source"
    images_base_folder = r"Z:\sequences"

    iou_thresh = 0.3
    class_aware = True
    max_age = -1
    minimum_confidence = 0.3
    tracker = gt.TrackerMode.HUNGARIAN
    max_center_distance = 0.2 #only used with TrackerMode.CENTER or TrackerMode.HUNGARIAN_CENTER

    # Drone pose-related paths (as in georeference_deepsort_mot)
    correction_folder = r"Z:\correction_data"
    additional_corrections_path = r"Z:\correction_data\corrections.json"
    apply_pose_smoothing = False  # set True if you also smoothed when georeferencing
    window_length = 11
    polyorder = 2

    show_live = True
    skip_existing = False
    target_base_folder = r"Z:\dets\georeferenced5\drawn2"
    create_video = True
    delete_images_after_video_creation = True

    if not show_live and not create_video:
        raise Exception("Nothing to do")

    config = {
        "iou_thresh": iou_thresh,
        "class_aware": class_aware,
        "max_age": max_age,
        "minimum_confidence": minimum_confidence,
        "tracker": str(tracker),
        "max_center_distance": max_center_distance,
        "apply_pose_smoothing": apply_pose_smoothing,
        "window_length": window_length,
        "polyorder": polyorder
    }

    if create_video:
        videowriter = FFMPEGWriter()
    else:
        videowriter = None

    # Load global additional corrections once
    all_additional_corrections = {}
    if os.path.exists(additional_corrections_path):
        with open(additional_corrections_path, "r", encoding="utf-8") as f:
            all_additional_corrections = json.load(f)
    all_corrections_dict = all_additional_corrections.get("corrections", {})

    track_files = {}
    detection_files = {}
    image_folders = {}

    # --- Collect georeferenced detection TXT files instead of CSV ---
    for root, dirs, files in os.walk(georef_dets_base_folder):
        for file in files:
            if not file.endswith(".txt"):
                continue
            # Z:\dets\georeferenced5\drawn\test

            if not file.startswith("26_3"):
                continue
            full_file_path = os.path.join(root, file)
            p = Path(full_file_path)
            if skip_existing and os.path.exists(os.path.join(target_base_folder, p.parent.name, p.stem)):
                continue
            key = p.parent.name + "/" + p.stem
            track_files[key] = full_file_path

    # --- Match detection files ---
    for root, dirs, files in os.walk(detections_base_folder):
        for file in files:
            full_file_path = os.path.join(root, file)
            p = Path(full_file_path)
            key = p.parent.name + "/" + p.stem
            if key in track_files:
                detection_files[key] = full_file_path

    # --- Match image folders ---
    for root, dirs, files in os.walk(images_base_folder):
        for d in dirs:
            full_dir_path = os.path.join(root, d)
            p = Path(full_dir_path)
            key = p.parent.name + "/" + p.stem
            if key in track_files:
                image_folders[key] = full_dir_path

    cv2.namedWindow("Tracks (image + global)", cv2.WINDOW_NORMAL)

    # --- Iterate all sequences (files) ---
    for seq_idx, (key, georef_txt_path) in enumerate(sorted(track_files.items())):
        print(f"\n=== Sequence {seq_idx + 1}/{len(track_files)}: {key}")
        print(f"Geo detections: {georef_txt_path}")

        # 1) LIVE: run tracking + build sequence_tracks/global_extent
        sequence_tracks, global_extent = load_tracks_live_from_georef_txt(
            georef_txt_path,
            iou_thresh=iou_thresh,
            class_aware=class_aware,
            max_age=max_age,
            tracker_mode=tracker,
            max_center_distance=max_center_distance,
            min_confidence=minimum_confidence,
        )

        cnt_tracks = defaultdict(int)
        for sequence_track in sequence_tracks.values():
            for entry in sequence_track:
                cnt_tracks[entry["tid"]] += 1
        config["tracks"] = cnt_tracks

        if not sequence_tracks or global_extent is None:
            print("  No tracks, skipping.")
            continue

        # 2) Load image-space detections
        det_path = detection_files.get(key)
        if det_path is None:
            print("  No detection file, skipping.")
            continue
        dets = read_detections_image_space(det_path)

        pixel_bboxes_by_row = {}
        for row_idx, det in enumerate(dets):
            _, x1, y1, x2, y2, conf, cls = det
            pixel_bboxes_by_row[row_idx] = (x1, y1, x2, y2, conf, cls)

        # 3) Find image folder
        img_root = image_folders.get(key)
        if img_root is None:
            print("  No image folder, skipping.")
            continue
        img_dir = os.path.join(img_root, "img1")
        if not os.path.isdir(img_dir):
            print(f"  No img1/ in {img_root}, skipping.")
            continue

        image_files = sorted(Path(img_dir).glob("*.jpg"))
        if not image_files:
            print("  No images, skipping.")
            continue

        # Frames used in this sequence (as zero-padded strings)
        frames_in_sequence = sorted(sequence_tracks.keys())

        # Flight id (parent id) is the prefix of the track filename BEFORE "_"
        # Example: 14_1.csv --> flight_id = "14"
        csv_path = Path(georef_txt_path)
        track_stem = csv_path.stem  # "14_1"
        flight_id = track_stem.split("_")[0]  # "14"

        # Additional frame-range corrections for this flight
        additional_corr_for_parent = all_corrections_dict.get(str(flight_id), [])

        with open(os.path.join(correction_folder, f"{flight_id}_dem_mesh_r2.json"), "r") as f:
            dem_meta = json.load(f)
        x_offset = dem_meta["origin"][0]
        y_offset = dem_meta["origin"][1]
        z_offset = dem_meta["origin"][2]

        # 4) Precompute drone positions for the frames we care about
        drone_positions = load_drone_positions_for_sequence(
            parent_id=flight_id,
            frame_ids_for_sequence=[img.stem for img in image_files],
            correction_folder=correction_folder,
            additional_corrections_for_parent=additional_corr_for_parent,
            apply_pose_smoothing=apply_pose_smoothing,
            window_length=window_length,
            polyorder=polyorder,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset
        )


        # Extend global_extent to also include drone positions (if any)
        if drone_positions:
            dx = [p[0] for p in drone_positions.values()]
            dy = [p[1] for p in drone_positions.values()]
            min_x, max_x, min_y, max_y = global_extent
            min_x = min(min_x, min(dx))
            max_x = max(max_x, max(dx))
            min_y = min(min_y, min(dy))
            max_y = max(max_y, max(dy))

            length = max_x - min_x
            height = max_y - min_y

            if length > height:
                max_y = min_y + length
            else:
                max_x = min_x + height

            global_extent = (min_x, max_x, min_y, max_y)

        canvas_cfg = make_global_canvas(global_extent)
        tracks_history = defaultdict(list)  # tid -> list of (cpx, cpy)
        drone_history = []  # list of (cpx, cpy)

        # For overlay text
        current_sub_folder = deviating_folders(images_base_folder, str(image_files[0]))

        target_image_files = []
        # --- Play frames live ---
        for img_file in image_files:
            frame_id = img_file.stem  # e.g. "00000001"
            img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # --- Left view: image with boxes in image space ---
            img_vis = img.copy()

            if frame_id in sequence_tracks:
                for tinfo in sequence_tracks[frame_id]:
                    row_idx = tinfo["source_id"] - 1
                    tid = tinfo["tid"]
                    conf = tinfo["conf"]
                    color = id_to_color(tid)

                    if row_idx in pixel_bboxes_by_row:
                        x1, y1, x2, y2, det_conf, cls = pixel_bboxes_by_row[row_idx]

                        if conf >= 0:
                            cv2.rectangle(
                                img_vis,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color,
                                thickness=2,
                            )
                        else:
                            draw_dashed_rectangle(
                                img_vis,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color,
                                2,
                                15,
                            )

                        cv2.putText(
                            img_vis,
                            f"ID {tid}",
                            (int(x1), max(0, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            # --- Right view: global space ---
            map_img = np.zeros(
                (canvas_cfg["height"], canvas_cfg["width"], 3), dtype=np.uint8
            )

            # Current frame's track boxes + track history
            if frame_id in sequence_tracks:
                for tinfo in sequence_tracks[frame_id]:
                    tid = tinfo["tid"]
                    color = id_to_color(tid)

                    gx1, gy1 = tinfo["gx1"], tinfo["gy1"]
                    gx2, gy2 = tinfo["gx2"], tinfo["gy2"]

                    px1, py1 = world_to_canvas(gx1, gy1, canvas_cfg)
                    px2, py2 = world_to_canvas(gx2, gy2, canvas_cfg)
                    cv2.rectangle(map_img, (px1, py1), (px2, py2), color, 2)

                    # center for history
                    cx = 0.5 * (gx1 + gx2)
                    cy = 0.5 * (gy1 + gy2)
                    cpx, cpy = world_to_canvas(cx, cy, canvas_cfg)
                    tracks_history[tid].append((cpx, cpy))

                    # NEW: write track id in global space
                    # cv2.putText(
                    #     map_img,
                    #     f"ID {tid}",
                    #     (px1, max(0, py1 +20)),  # a bit above the box
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     color,
                    #     1,
                    #     cv2.LINE_AA,
                    # )

            for tid, pts in tracks_history.items():
                color = id_to_color(tid)
                for i in range(1, len(pts)):
                    cv2.line(map_img, pts[i - 1], pts[i], color, 1)

                # label at last point of the path
                if pts:
                    lx, ly = pts[-1]
                    cv2.putText(
                        map_img,
                        f"ID {tid}",
                        (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            # Drone position + path
            if frame_id in drone_positions:
                dx, dy, dz = drone_positions[frame_id]
                dp_x, dp_y = world_to_canvas(dx, dy, canvas_cfg)
                drone_history.append((dp_x, dp_y))

                # Draw drone path
                for i in range(1, len(drone_history)):
                    cv2.line(map_img, drone_history[i - 1], drone_history[i], (0, 255, 255), 2)

                # Draw current drone position
                cv2.circle(map_img, (dp_x, dp_y), 6, (0, 255, 255), -1)
                cv2.putText(
                    map_img,
                    "Drone",
                    (dp_x + 8, dp_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Axes
            draw_axes_on_canvas(map_img, canvas_cfg, num_ticks=4)

            # --- Combine views ---
            h_img, w_img = img_vis.shape[:2]
            h_map, w_map = map_img.shape[:2]

            if h_map != h_img:
                scale = h_img / h_map
                new_w = int(w_map * scale)
                map_resized = cv2.resize(map_img, (new_w, h_img))
            else:
                map_resized = map_img

            combined = np.hstack([img_vis, map_resized])

            cv2.putText(
                combined,
                f"{key} | {frame_id}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if show_live:
                cv2.imshow("Tracks (image + global)", combined)

                # Controls: q/ESC = quit, n = next sequence, space = pause
                key_press = cv2.waitKey(30) & 0xFF
                if key_press in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    raise SystemExit
                if key_press == ord("n"):
                    print("  -> Skipping to next sequence.")
                    break
                if key_press == ord(" "):
                    # pause until space / n / q / ESC
                    while True:
                        key2 = cv2.waitKey(0) & 0xFF
                        if key2 in (ord("q"), 27):
                            cv2.destroyAllWindows()
                            raise SystemExit
                        if key2 == ord("n"):
                            print("  -> Skipping to next sequence.")
                            break
                        if key2 == ord(" "):
                            break
                    if key2 == ord("n"):
                        break

            if create_video:
                image_target_folder = os.path.join(target_base_folder, Path(current_sub_folder))
                image_target_file = os.path.join(image_target_folder, frame_id + ".jpg")
                target_image_files.append(image_target_file)
                os.makedirs(image_target_folder, exist_ok=True)
                cv2.imwrite(image_target_file, combined)

        if create_video and len(target_image_files) > 0 and current_sub_folder is not None:
            with open(os.path.join(target_base_folder, Path(current_sub_folder), "config.json"), "w") as config_file:
                json.dump(config, config_file)
            p = Path(georef_txt_path)
            video_path = os.path.join(target_base_folder, Path(current_sub_folder), f"{p.stem}.mp4")
            gen = ((idx, cv2.imread(x)) for (idx, x) in enumerate(target_image_files))
            videowriter.write(video_path, gen)
            print(f"Created video {video_path}")
            if delete_images_after_video_creation:
                for target_image_file in target_image_files:
                    os.remove(target_image_file)
    cv2.destroyAllWindows()
    print("Done.")
