from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from alfspy.core.convert import world_to_pixel_coord
from alfspy.core.convert.convert import world_to_pixel_coord2
from alfspy.core.rendering import Resolution
from alfspy.core.util.geo import get_aabb
from alfspy.orthografic_projection import get_camera_for_frame
from alfspy.render.render import release_all, read_gltf, process_render_data, make_mgl_context
from pyproj import Transformer, CRS
from pyproj.enums import TransformDirection
from pyrr import Vector3
from trimesh import Trimesh

from src.bambi.georeference_deepsort_mot import deviating_folders
from src.bambi.georeferenced_tracking import Detection
from src.bambi.util.projection_util import label_to_world_coordinates

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
import csv
import math
import pathlib

@dataclass
class Track:
    track_id: int
    # list of (frame_id, detection), sorted by frame_id
    items: List[Tuple[int, Detection]]

    @property
    def frames(self) -> List[int]:
        return [f for f, _ in self.items]

    @property
    def detections(self) -> List[Detection]:
        return [d for _, d in self.items]

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def interpolate_gaps_for_track(
    items: List[Tuple[int, Detection]],
    max_gap: Optional[int] = None,
    interpolate_conf: bool = False,
) -> Tuple[List[Tuple[int, Detection]], Set[int]]:
    """
    Given a single track as [(frame, Detection), ...] sorted by frame,
    insert linearly interpolated detections for missing frames between
    consecutive known frames. Does NOT extrapolate before the first or after
    the last known detection.

    Args:
        items: sorted list of (frame, Detection)
        max_gap: if set, only interpolate gaps with (end - start) - 1 <= max_gap
                 (i.e., number of missing frames <= max_gap)
        interpolate_conf: if True, linearly interpolate confidence; otherwise
                          use min(left_conf, right_conf) for the gap.

    Returns:
        (new_items, interpolated_frames)
        new_items: sorted list including originals and inserted detections
        interpolated_frames: set of frame ids that were created by interpolation
    """
    if not items:
        return [], set()

    # Ensure sorted
    items = sorted(items, key=lambda t: t[0])
    out: List[Tuple[int, Detection]] = []
    interpolated_frames: Set[int] = set()

    for i, (f0, d0) in enumerate(items):
        out.append((f0, d0))
        if i == len(items) - 1:
            break  # no next anchor

        f1, d1 = items[i + 1]
        gap = f1 - f0 - 1
        if gap <= 0:
            continue
        if max_gap is not None and gap > max_gap:
            # too large to interpolate -> skip
            continue

        denom = float(f1 - f0)
        for f in range(f0 + 1, f1):
            t = (f - f0) / denom  # distance weighting
            x1 = _lerp(d0.x1, d1.x1, t)
            y1 = _lerp(d0.y1, d1.y1, t)
            z1 = _lerp(d0.z1, d1.z1, t)
            x2 = _lerp(d0.x2, d1.x2, t)
            y2 = _lerp(d0.y2, d1.y2, t)
            z2 = _lerp(d0.z2, d1.z2, t)
            if interpolate_conf:
                conf = _lerp(d0.conf, d1.conf, t)
            else:
                conf = -1 # signale that it is a interpolated confidence
                # conf = min(d0.conf, d1.conf)
            # Class: stick with the left anchor (common for MOT)
            cls = d0.cls

            out.append((f, Detection(f, x1, y1, z1, x2, y2, z2, conf, cls)))
            interpolated_frames.add(f)

    # Sort again because we inserted in-between frames
    out.sort(key=lambda t: t[0])
    return out, interpolated_frames

def interpolate_gaps(
    tracks: Dict[int, List[Tuple[int, Detection]]],
    max_gap: Optional[int] = None,
    interpolate_conf: bool = False,
) -> Tuple[Dict[int, List[Tuple[int, Detection]]], Dict[int, Set[int]]]:
    """
    Apply interpolation to all tracks.

    Returns:
        (filled_tracks, interp_index)
        filled_tracks: same shape as input but with missing frames inserted
        interp_index: track_id -> set of frame_ids that were interpolated
    """
    filled: Dict[int, List[Tuple[int, Detection]]] = {}
    idx: Dict[int, Set[int]] = {}

    for tid, items in tracks.items():
        new_items, interped = interpolate_gaps_for_track(
            items, max_gap=max_gap, interpolate_conf=interpolate_conf
        )
        filled[tid] = new_items
        idx[tid] = interped
    return filled, idx

def _maybe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None

def read_tracks_csv(path: str) -> Dict[int, List[Tuple[int, Detection]]]:
    """
    Reads a MOT-like tracks CSV:
        frame_id,track_id,x1,y1,x2,y2,confidence,class_id
    Returns:
        dict: track_id -> list of (frame_id, Detection), sorted by frame_id
    """
    tracks: Dict[int, List[Tuple[int, Detection]]] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # Detect and skip a header row if present
            # (e.g., "frame_id, track_id, ...")
            first_as_int = _maybe_int(row[0].strip())
            if first_as_int is None:
                # Likely a header; skip
                continue

            frame_id = first_as_int
            track_id = int(row[1].strip())

            x1 = float(row[2])
            y1 = float(row[3])
            z1 = float(row[4])
            x2 = float(row[5])
            y2 = float(row[6])
            z2 = float(row[7])
            conf = float(row[8])
            cls = int(float(row[9]))  # tolerant if written as "0.0"

            det = Detection(frame=frame_id, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2, conf=conf, cls=cls)
            tracks.setdefault(track_id, []).append((frame_id, det))

    # Sort each track by frame_id
    for tid in tracks:
        tracks[tid].sort(key=lambda t: t[0])

    return tracks

def read_tracks_csv_as_list(path: str | pathlib.Path) -> List[Track]:
    """Same as read_tracks_csv but returns a list of Track objects sorted by track_id."""
    d = read_tracks_csv(path)
    return [Track(track_id=tid, items=items) for tid, items in sorted(d.items(), key=lambda x: x[0])]

# Optional helpers:

def iter_frame_index(tracks: Dict[int, List[Tuple[int, Detection]]]) -> Dict[int, List[Tuple[int, Detection]]]:
    """
    Build a frame index: frame_id -> list of (track_id, Detection).
    Useful if you want to iterate frame-by-frame across all tracks.
    """
    frame_index: Dict[int, List[Tuple[int, Detection]]] = {}
    for tid, items in tracks.items():
        for frame_id, det in items:
            frame_index.setdefault(frame_id, []).append((tid, det))
    # Sort detections per frame by track_id for stable order
    for fr in frame_index:
        frame_index[fr].sort(key=lambda td: td[0])
    return frame_index


if __name__ == '__main__':
    # Paths
    source_detection_base = r"Z:\dets\source"
    georeferenced_tracks_base = r"Z:\dets\georeferenced_tracks"
    correction_folder = r"Z:\correction_data"
    target_base = r"Z:\dets\georeferenced_tracks_extended"
    additional_corrections_path = r"Z:\correction_data\corrections.json"
    input_resolution = Resolution(1024, 1024)
    skip_existing = True
    rel_transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32633))
    transform_to_target_crs = False
    write_global_coordinates = False

    ##################################################################################

    with open(additional_corrections_path) as f:
        all_additional_corrections = json.load(f)

    # Dictionary parent_id -> list of files
    parent_dict = defaultdict(list)

    # Loop through both directories
    for root, dirs, files in os.walk(georeferenced_tracks_base):
        for file in files:
            if file.endswith(".csv") and "_" in file:
                parent_id = file.split("_")[0]  # Extract parent ID before '_'
                full_path = os.path.join(root, file)
                parent_dict[parent_id].append(full_path)

    # Convert defaultdict to normal dict (optional)
    parent_dict = dict(parent_dict)

    number_of_flights = len(parent_dict)
    transformation_errors = 0
    total_bb = 0
    for idx, (parent, files) in enumerate(parent_dict.items()):
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
            if not write_global_coordinates:
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

            for f in files:
                p = Path(f)
                target_folder = os.path.join(target_base, deviating_folders(georeferenced_tracks_base, f))
                os.makedirs(target_folder, exist_ok=True)
                target_file = os.path.join(target_folder, p.name)
                if skip_existing and os.path.exists(target_file):
                    continue
                tracks = read_tracks_csv(f)
                filled_tracks, interp_index = interpolate_gaps(tracks, max_gap=None, interpolate_conf=False)
                to_write = defaultdict(list)
                for track_idx, track in filled_tracks.items():
                    for frame_idx, det in track:
                        x1 = det.x1
                        y1 = det.y1
                        z1 = det.z1
                        x2 = det.x2
                        y2 = det.y2
                        z2 = det.z2
                        if write_global_coordinates:
                            to_write[frame_idx].append((track_idx, x1, y1, z1, x2, y2, z2, det.conf, det.cls))
                        else:
                            for additional_correction in additional_corrections:
                                if additional_correction["start frame"] < frame_idx < additional_correction[
                                    "end frame"]:
                                    translation = additional_correction.get('translation', translation)
                                    rotation = additional_correction.get('rotation', rotation)
                                    break
                            x1 -= x_offset
                            y1 -= y_offset
                            z1 -= z_offset
                            x2 -= x_offset
                            y2 -= y_offset
                            z2 -= z_offset
                            image_metadata = poses["images"][frame_idx]
                            fov = image_metadata["fovy"][0]

                            # project labels
                            cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
                            cor_translation = Vector3([translation['x'], translation['y'], translation['z']],
                                                      dtype='f4')
                            camera = get_camera_for_frame(poses, frame_idx, cor_rotation_eulers, cor_translation)
                            pxl_coord = np.asarray(world_to_pixel_coord2([x1, x2, x2, x2],[y1, y1, y2, y2],[z1, z1, z2, z2], input_resolution.width, input_resolution.height, camera))
                            if None in pxl_coord:
                                continue
                            # todo error maybe due to missing altitude?
                            xx = pxl_coord[:,0]
                            yy = pxl_coord[:,1]
                            minx = min(xx)
                            maxx = max(xx)
                            miny = min(yy)
                            maxy = max(yy)
                            to_write[frame_idx].append((track_idx, minx, miny, maxx, maxy, det.conf, det.cls))

                with open(target_file, "w") as f:
                    for frame_idx in sorted(to_write.keys()):
                        for det in to_write[frame_idx]:
                            if len(det) == 7:
                                t_id = det[0]
                                x1 = det[1]
                                y1 = det[2]
                                x2 = det[3]
                                y2 = det[4]
                                conf = det[5]
                                cls = det[6]
                                f.write(f"{frame_idx:08d},{t_id},{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f},{conf:.6f},{cls}\n")
                            elif len(det) == 9:
                                t_id = det[0]
                                x1 = det[1]
                                y1 = det[2]
                                z1 = det[3]
                                x2 = det[4]
                                y2 = det[5]
                                z2 = det[6]
                                conf = det[7]
                                cls = det[8]
                                f.write(f"{frame_idx:08d},{t_id},{x1:.6f},{y1:.6f},{z1:.6f},{x2:.6f},{y2:.6f},{z2:.6f},{conf:.6f},{cls}\n")
                            else:
                                raise Exception()

        finally:
            # free up resources
            release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
    print(f"Could not transform {transformation_errors} of {total_bb} bounding boxes")