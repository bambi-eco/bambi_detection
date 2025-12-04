import colorsys
import hashlib
import json
import os
import requests
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from scipy.signal import savgol_filter
from pyproj import CRS, Transformer

import src.bambi.georeferenced_tracking as gt
from src.bambi.georeference_deepsort_mot import deviating_folders
from src.bambi.video.video_writer import FFMPEGWriter
import math


# ============================================================
# 1. CORE TRACKING & INTERPOLATION
# ============================================================

def build_sequence_tracks_from_results(results):
    sequence_tracks = defaultdict(list)
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    for row_idx, (frame, tid, d) in enumerate(results):
        frame_str = f"{frame:08d}"
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2

        min_x = min(min_x, x1, x2)
        max_x = max(max_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)

        sequence_tracks[frame_str].append({
            "row_idx": row_idx,
            "tid": tid,
            "conf": d.conf,
            "cls": d.cls,
            "gx1": x1, "gy1": y1, "gx2": x2, "gy2": y2,
            "source_id": d.source_id,
            "interpolated": d.interpolated
        })

    if min_x == float("inf"): return {}, None
    return sequence_tracks, (min_x, max_x, min_y, max_y)


def load_tracks_live_from_georef_txt(
        georef_txt_path, iou_thresh=0.3, class_aware=True, max_age=-1,
        tracker_mode=gt.TrackerMode.HUNGARIAN, max_center_distance=0.2, min_confidence=0.0
):
    frames = gt.read_detections(georef_txt_path, min_confidence=min_confidence)
    results = gt.track_detections(
        frames, iou_thr=iou_thresh, class_aware=class_aware, max_age=max_age,
        tracker_mode=tracker_mode, max_center_distance=max_center_distance
    )
    return build_sequence_tracks_from_results(results)


def interpolate_global_and_local_tracks(sequence_tracks, pixel_bboxes_by_row):
    tracks = defaultdict(dict)
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    for frame_str, track_list in sequence_tracks.items():
        frame_idx = int(frame_str)
        for t in track_list:
            tid = t["tid"]
            row_idx = t.get("source_id", -1) - 1

            if row_idx in pixel_bboxes_by_row:
                lx1, ly1, lx2, ly2, _, _ = pixel_bboxes_by_row[row_idx]
            else:
                lx1, ly1, lx2, ly2 = 0, 0, 0, 0

            tracks[tid][frame_idx] = {
                "frame": frame_idx,
                "tid": tid,
                "cls": t["cls"],
                "gx1": t["gx1"], "gy1": t["gy1"], "gx2": t["gx2"], "gy2": t["gy2"],
                "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                "interpolated": t["interpolated"]
            }

    densified_tracks = defaultdict(list)

    for tid, frames_dict in tracks.items():
        sorted_frames = sorted(frames_dict.keys())
        if not sorted_frames: continue

        min_f, max_f = sorted_frames[0], sorted_frames[-1]

        for f in range(min_f, max_f + 1):
            frame_str = f"{f:08d}"

            if f in frames_dict:
                d = frames_dict[f]
                min_x = min(min_x, d["gx1"], d["gx2"])
                max_x = max(max_x, d["gx1"], d["gx2"])
                min_y = min(min_y, d["gy1"], d["gy2"])
                max_y = max(max_y, d["gy1"], d["gy2"])
                densified_tracks[frame_str].append(d)
            else:
                prev_f = max([k for k in sorted_frames if k < f])
                next_f = min([k for k in sorted_frames if k > f])
                start, end = frames_dict[prev_f], frames_dict[next_f]

                ratio = (f - prev_f) / (next_f - prev_f)

                gx1 = start["gx1"] + (end["gx1"] - start["gx1"]) * ratio
                gy1 = start["gy1"] + (end["gy1"] - start["gy1"]) * ratio
                gx2 = start["gx2"] + (end["gx2"] - start["gx2"]) * ratio
                gy2 = start["gy2"] + (end["gy2"] - start["gy2"]) * ratio

                lx1 = start["lx1"] + (end["lx1"] - start["lx1"]) * ratio
                ly1 = start["ly1"] + (end["ly1"] - start["ly1"]) * ratio
                lx2 = start["lx2"] + (end["lx2"] - start["lx2"]) * ratio
                ly2 = start["ly2"] + (end["ly2"] - start["ly2"]) * ratio

                min_x = min(min_x, gx1, gx2)
                max_x = max(max_x, gx1, gx2)
                min_y = min(min_y, gy1, gy2)
                max_y = max(max_y, gy1, gy2)

                densified_tracks[frame_str].append({
                    "frame": f, "tid": tid, "cls": start["cls"],
                    "gx1": gx1, "gy1": gy1, "gx2": gx2, "gy2": gy2,
                    "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                    "interpolated": 1
                })

    if min_x == float("inf"): return {}, None
    return densified_tracks, (min_x, max_x, min_y, max_y)


# ============================================================
# 2. COORDINATE & CANVAS HELPERS
# ============================================================

def id_to_color(identifier, saturation=0.65, lightness=0.5):
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2 ** 32
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def pad_extent_to_match_aspect_ratio(extent, width, height, margin):
    min_x, max_x, min_y, max_y = extent
    draw_w = width - 2 * margin
    draw_h = height - 2 * margin
    target_ar = draw_w / draw_h

    data_w = max_x - min_x
    data_h = max_y - min_y
    data_ar = data_w / data_h if data_h > 0 else 1.0

    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    if data_ar > target_ar:
        new_h = data_w / target_ar
        min_y = cy - new_h / 2
        max_y = cy + new_h / 2
    else:
        new_w = data_h * target_ar
        min_x = cx - new_w / 2
        max_x = cx + new_w / 2

    return (min_x, max_x, min_y, max_y)


def make_global_canvas(global_extent, width=800, height=800, margin=60):
    min_x, max_x, min_y, max_y = global_extent
    span_x = max(max_x - min_x, 1e-6)
    scale = (width - 2 * margin) / span_x
    return {
        "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
        "scale": scale, "margin": margin, "width": width, "height": height,
    }


def world_to_canvas(x, y, canvas_cfg):
    min_x = canvas_cfg["min_x"]
    min_y = canvas_cfg["min_y"]
    scale = canvas_cfg["scale"]
    margin = canvas_cfg["margin"]
    height = canvas_cfg["height"]
    px = int(margin + (x - min_x) * scale)
    py = int(height - (margin + (y - min_y) * scale))
    return px, py


def draw_axes_on_canvas(map_img, canvas_cfg, num_ticks=4):
    min_x, max_x = canvas_cfg["min_x"], canvas_cfg["max_x"]
    min_y, max_y = canvas_cfg["min_y"], canvas_cfg["max_y"]
    axis_color = (200, 200, 200)
    text_color = (255, 255, 255)

    bl = world_to_canvas(min_x, min_y, canvas_cfg)
    br = world_to_canvas(max_x, min_y, canvas_cfg)
    tl = world_to_canvas(min_x, max_y, canvas_cfg)

    cv2.line(map_img, bl, br, axis_color, 1)
    cv2.line(map_img, bl, tl, axis_color, 1)

    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        val = min_x + t * (max_x - min_x)
        px, py = world_to_canvas(val, min_y, canvas_cfg)
        cv2.line(map_img, (px, py), (px, py + 5), axis_color, 1)
        cv2.putText(map_img, f"{int(val)}", (px - 20, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        val = min_y + t * (max_y - min_y)
        px, py = world_to_canvas(min_x, val, canvas_cfg)
        cv2.line(map_img, (px - 5, py), (px, py), axis_color, 1)
        cv2.putText(map_img, f"{int(val)}", (px - 60, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    h, w = map_img.shape[:2]
    cv2.putText(map_img, "Easting (X)", (w // 2, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(map_img, "Northing (Y)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


# ============================================================
# 3. MAP TILE PROVIDER
# ============================================================

class MapTileProvider:
    OPENSTREETMAP = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    ESRI_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    def __init__(self, tile_url=None, cache_dir=None, utm_epsg=32633):
        self.tile_url = tile_url or self.OPENSTREETMAP
        self.cache_dir = cache_dir
        self.transformer = Transformer.from_crs(CRS.from_epsg(utm_epsg), CRS.from_epsg(4326), always_xy=True)
        self.headers = {'User-Agent': 'VisScript/1.0'}
        if cache_dir: os.makedirs(cache_dir, exist_ok=True)

    def utm_to_latlon(self, x, y):
        return self.transformer.transform(x, y)[::-1]

    def latlon_to_tile(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile

    def tile_to_latlon(self, x, y, zoom):
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad), lon

    def download_tile(self, x, y, zoom):
        cache_path = None
        if self.cache_dir:
            h = hashlib.md5(self.tile_url.encode()).hexdigest()[:8]
            cache_path = os.path.join(self.cache_dir, f"{h}_{zoom}_{x}_{y}.png")
            if os.path.exists(cache_path):
                return cv2.imread(cache_path)

        url = self.tile_url.format(z=zoom, x=x, y=y)
        try:
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if self.cache_dir and img is not None: cv2.imwrite(cache_path, img)
                return img
        except:
            pass
        return None

    def get_map_background(self, global_extent, canvas_cfg):
        min_x, max_x, min_y, max_y = global_extent
        min_lat, min_lon = self.utm_to_latlon(min_x, min_y)
        max_lat, max_lon = self.utm_to_latlon(max_x, max_y)
        if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
        if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon

        cw, ch = canvas_cfg["width"], canvas_cfg["height"]
        zoom = 18
        for z in range(19, 12, -1):
            x1, y1 = self.latlon_to_tile(max_lat, min_lon, z)
            x2, y2 = self.latlon_to_tile(min_lat, max_lon, z)
            if (abs(x2 - x1) + 1) * 256 > cw and (abs(y2 - y1) + 1) * 256 > ch:
                zoom = z
                break

        tx1, ty1 = self.latlon_to_tile(max_lat, min_lon, zoom)
        tx2, ty2 = self.latlon_to_tile(min_lat, max_lon, zoom)

        stitch_w = (tx2 - tx1 + 1) * 256
        stitch_h = (ty2 - ty1 + 1) * 256
        stitch = np.zeros((stitch_h, stitch_w, 3), dtype=np.uint8)

        for y in range(ty1, ty2 + 1):
            for x in range(tx1, tx2 + 1):
                t = self.download_tile(x, y, zoom)
                if t is not None:
                    py, px = (y - ty1) * 256, (x - tx1) * 256
                    stitch[py:py + 256, px:px + 256] = t

        top_lat, left_lon = self.tile_to_latlon(tx1, ty1, zoom)
        btm_lat, rgt_lon = self.tile_to_latlon(tx2 + 1, ty2 + 1, zoom)

        def ll2px(lat, lon):
            px = (lon - left_lon) / (rgt_lon - left_lon) * stitch_w
            py = (top_lat - lat) / (top_lat - btm_lat) * stitch_h
            return int(px), int(py)

        px1, py1 = ll2px(max_lat, min_lon)
        px2, py2 = ll2px(min_lat, max_lon)

        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(stitch_w, px2), min(stitch_h, py2)

        if px2 <= px1 or py2 <= py1: return None
        crop = stitch[py1:py2, px1:px2]

        iw, ih = cw - 2 * canvas_cfg["margin"], ch - 2 * canvas_cfg["margin"]
        final = np.zeros((ch, cw, 3), dtype=np.uint8)
        try:
            resized_crop = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA)
            final[canvas_cfg["margin"]:canvas_cfg["margin"] + ih,
            canvas_cfg["margin"]:canvas_cfg["margin"] + iw] = resized_crop
        except Exception as e:
            print(f"Map resize error: {e}")
            return None

        return final


# ============================================================
# 4. DATA LOADING HELPERS
# ============================================================

def read_detections_image_space(det_path):
    dets = []
    if not os.path.exists(det_path): return dets
    with open(det_path, "r", encoding="utf-8") as source:
        for idx, line in enumerate(source):
            if idx == 0: continue
            parts = line.split()
            if len(parts) < 7: continue
            dets.append((parts[0], float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4]), float(parts[5]), int(parts[6])))
    return dets


def load_drone_positions_for_sequence(parent_id, frame_ids, correction_folder,
                                      additional_corrs, offset=(0, 0, 0)):
    path = os.path.join(correction_folder, f"{parent_id}_matched_poses.json")
    corr_path = os.path.join(correction_folder, f"{parent_id}_correction.json")
    if not os.path.exists(path): return {}

    with open(path, "r") as f:
        poses = json.load(f)

    base_t = {"x": 0.0, "y": 0.0, "z": 0.0}
    if os.path.exists(corr_path):
        with open(corr_path, "r") as f: base_t = json.load(f).get("translation", base_t)

    images = poses.get("images", [])
    drone_pos = {}

    for fid in frame_ids:
        try:
            idx = int(fid)
        except:
            continue
        if not (0 <= idx < len(images)): continue

        t = base_t
        for ac in additional_corrs:
            if ac["start frame"] < idx < ac["end frame"]:
                t = ac.get("translation", t)

        loc = np.array(images[idx]["location"])
        off = np.array([t["x"] + offset[0], t["y"] + offset[1], t["z"] + offset[2]])
        drone_pos[fid] = loc + off
    return drone_pos


def load_fov_polygons(path):
    polys = {}
    if not os.path.exists(path): return polys
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if len(parts) < 2: continue
            fid = int(parts[0])
            n = int(parts[1])
            pts = []
            for i in range(n):
                idx = 2 + i * 3
                if idx + 2 < len(parts):
                    pts.append((float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])))
            polys[fid] = pts
    return polys


def draw_fov_polygon(img, pts, cfg, color, thick=2, alpha=0.15):
    if not pts: return
    cpts = np.array([world_to_canvas(p[0], p[1], cfg) for p in pts], dtype=np.int32)
    if alpha > 0:
        ov = img.copy()
        cv2.fillPoly(ov, [cpts], color)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [cpts], True, color, thick)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    georef_dets_base = r"Z:\dets\georeferenced5"
    dets_base = r"Z:\dets\source"
    img_base = r"Z:\sequences"
    target_base = r"Z:\dets\georeferenced5\drawn_interpolated"

    correction_folder = r"Z:\correction_data"
    add_corr_path = r"Z:\correction_data\corrections.json"
    fov_folder = r"Z:\dets\georeferenced_fov"

    iou_thresh = 0.3
    show_live = True
    create_video = True
    show_map = True
    show_fov = True

    map_tile_url = MapTileProvider.ESRI_SATELLITE
    map_cache = r"Z:\map_tile_cache"

    if create_video: writer = FFMPEGWriter()

    all_corrs = {}
    if os.path.exists(add_corr_path):
        with open(add_corr_path, "r") as f: all_corrs = json.load(f).get("corrections", {})

    track_files = {}
    for root, _, files in os.walk(georef_dets_base):
        for f in files:
            if f.endswith(".txt") and f.startswith("14_1"):
                track_files[f"{Path(root).name}/{Path(f).stem}"] = os.path.join(root, f)

    cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)

    # --- PROCESS SEQUENCES ---
    for seq_idx, (key, georef_path) in enumerate(sorted(track_files.items())):
        print(f"Processing {key}...")

        # 1. Track
        seq_tracks_raw, glob_extent = load_tracks_live_from_georef_txt(
            georef_path, iou_thresh=iou_thresh
        )
        if not seq_tracks_raw: continue

        # 2. Match Local Detections
        det_file = None
        for root, _, files in os.walk(dets_base):
            for f in files:
                if f"{Path(root).name}/{Path(f).stem}" == key:
                    det_file = os.path.join(root, f)
                    break

        pix_bboxes = {}
        if det_file:
            raw_dets = read_detections_image_space(det_file)
            for idx, d in enumerate(raw_dets):
                pix_bboxes[idx] = (d[1], d[2], d[3], d[4], d[5], d[6])

        # 3. Interpolate
        final_tracks, glob_extent = interpolate_global_and_local_tracks(seq_tracks_raw, pix_bboxes)
        if not final_tracks: continue

        # 4. Setup Paths & Poses
        img_dir = None
        for root, dirs, _ in os.walk(img_base):
            for d in dirs:
                if f"{Path(root).name}/{d}" == key:
                    img_dir = os.path.join(root, d, "img1")

        if not img_dir: continue
        img_files = sorted(Path(img_dir).glob("*.jpg"))
        if not img_files: continue
        img_map = {f.stem: f for f in img_files}

        all_frames_int = sorted(set([int(k) for k in final_tracks.keys()] + [int(k) for k in img_map.keys()]))
        start_f, end_f = all_frames_int[0], all_frames_int[-1]

        flight_id = Path(georef_path).stem.split("_")[0]
        dem_path = os.path.join(correction_folder, f"{flight_id}_dem_mesh_r2.json")
        if not os.path.exists(dem_path): continue
        with open(dem_path) as f:
            dem = json.load(f)

        render_frames = [f"{i:08d}" for i in range(start_f, end_f + 1) if f"{i:08d}" in img_map]

        drone_pos = load_drone_positions_for_sequence(
            flight_id, render_frames, correction_folder, all_corrs.get(flight_id, []),
            offset=dem["origin"]
        )

        # 5. Finalize Extent & Map
        dx = [p[0] for p in drone_pos.values()]
        dy = [p[1] for p in drone_pos.values()]
        if dx:
            glob_extent = (
                min(glob_extent[0], min(dx)), max(glob_extent[1], max(dx)),
                min(glob_extent[2], min(dy)), max(glob_extent[3], max(dy))
            )

        padded_extent = pad_extent_to_match_aspect_ratio(glob_extent, 800, 800, 60)
        canvas_cfg = make_global_canvas(padded_extent, 800, 800, 60)

        map_bg = None
        if show_map:
            prov = MapTileProvider(map_tile_url, map_cache)
            map_bg = prov.get_map_background(padded_extent, canvas_cfg)
            if map_bg is not None: map_bg = (map_bg * 0.4).astype(np.uint8)

        fov_polys = {}
        if show_fov:
            fov_polys = load_fov_polygons(os.path.join(fov_folder, f"{flight_id}_fov_georeferenced.txt"))

        track_history = defaultdict(list)
        drone_history = []
        vid_frames = []

        # 6. Render
        for frame_id in render_frames:
            img_vis = cv2.imread(str(img_map[frame_id]))
            if img_vis is None: continue

            if map_bg is not None:
                map_img = map_bg.copy()
            else:
                map_img = np.zeros((canvas_cfg["height"], canvas_cfg["width"], 3), dtype=np.uint8)

            if frame_id in drone_pos and int(frame_id) in fov_polys:
                draw_fov_polygon(map_img, fov_polys[int(frame_id)], canvas_cfg, (0, 200, 200))

            # TRACK VISIBLE IDs in this frame
            visible_tids = set()

            if frame_id in final_tracks:
                for t in final_tracks[frame_id]:
                    tid, interp = t["tid"], t["interpolated"]
                    visible_tids.add(tid)  # Mark as visible
                    color = id_to_color(tid)

                    label = f"ID {tid}"

                    # Image Space
                    lx1, ly1, lx2, ly2 = int(t["lx1"]), int(t["ly1"]), int(t["lx2"]), int(t["ly2"])
                    if interp:
                        draw_dashed_rectangle(img_vis, (lx1, ly1), (lx2, ly2), color, 2, 8)
                    else:
                        cv2.rectangle(img_vis, (lx1, ly1), (lx2, ly2), color, 2)

                    cv2.putText(img_vis, label, (lx1, max(0, ly1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)

                    # Global Space
                    gx1, gy1, gx2, gy2 = t["gx1"], t["gy1"], t["gx2"], t["gy2"]
                    px1, py1 = world_to_canvas(gx1, gy1, canvas_cfg)
                    px2, py2 = world_to_canvas(gx2, gy2, canvas_cfg)

                    if interp:
                        draw_dashed_rectangle(map_img, (px1, py1), (px2, py2), color, 2, 6)
                    else:
                        cv2.rectangle(map_img, (px1, py1), (px2, py2), color, 2)

                    # Label on global box
                    min_py = min(py1, py2)
                    cv2.putText(map_img, label, (px1, max(0, min_py - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)

                    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                    cpx, cpy = world_to_canvas(cx, cy, canvas_cfg)
                    track_history[tid].append((cpx, cpy))

            # History Loop - ONLY draw label if NOT visible in current frame
            for tid, pts in track_history.items():
                color = id_to_color(tid)
                if len(pts) > 1:
                    cv2.polylines(map_img, [np.array(pts)], False, color, 1)

                # Draw ID label at the last known position IF NOT visible now
                if tid not in visible_tids and pts:
                    lx, ly = pts[-1]
                    cv2.putText(map_img, f"ID {tid}", (lx + 5, ly - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if frame_id in drone_pos:
                dpos = drone_pos[frame_id]
                dcx, dcy = world_to_canvas(dpos[0], dpos[1], canvas_cfg)
                drone_history.append((dcx, dcy))
                if len(drone_history) > 1:
                    cv2.polylines(map_img, [np.array(drone_history)], False, (0, 255, 255), 2)
                cv2.circle(map_img, (dcx, dcy), 6, (0, 255, 255), -1)

            draw_axes_on_canvas(map_img, canvas_cfg)

            h_img, w_img = img_vis.shape[:2]
            h_map, w_map = map_img.shape[:2]
            scale = h_img / h_map
            map_resized = cv2.resize(map_img, (int(w_map * scale), h_img))
            combined = np.hstack([img_vis, map_resized])

            cv2.putText(combined, f"{key} | {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if show_live:
                cv2.imshow("Vis", combined)
                k = cv2.waitKey(1)
                if k == 27 or k == ord('q'): exit()

            if create_video:
                tmp_path = os.path.join(target_base, "tmp", f"{frame_id}.jpg")
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                cv2.imwrite(tmp_path, combined)
                vid_frames.append(tmp_path)

        if create_video and vid_frames:
            sub = deviating_folders(img_base, str(img_files[0]))
            out_dir = os.path.join(target_base, sub)
            os.makedirs(out_dir, exist_ok=True)
            vpath = os.path.join(out_dir, f"{Path(georef_path).stem}.mp4")
            gen = ((i, cv2.imread(f)) for i, f in enumerate(vid_frames))
            writer.write(vpath, gen)
            for f in vid_frames: os.remove(f)

    cv2.destroyAllWindows()