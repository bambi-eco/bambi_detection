import colorsys
import csv
import hashlib
import json
import os
import requests
import io
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from scipy.signal import savgol_filter
from pyproj import CRS, Transformer

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
        interpolated = d.interpolated

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
                "source_id": source_id,
                "interpolated": interpolated
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
    hue = int.from_bytes(h[:4], "big") / 2 ** 32  # [0,1)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2

    # Ensure coordinates are ordered correctly (min to max)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

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
# FOV Polygon loading and drawing
# ============================================================

def load_fov_polygons(fov_file_path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    Load georeferenced FOV polygons from file.

    File format (from georeference_deepsort_mot.py):
      # Header comments starting with #
      frame_idx num_points x1 y1 z1 x2 y2 z2 ...

    Returns:
      dict[frame_idx] -> list of (x, y, z) world coordinates
    """
    fov_polygons = {}

    if not os.path.exists(fov_file_path):
        return fov_polygons

    with open(fov_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            frame_idx = int(parts[0])
            num_points = int(parts[1])

            if num_points == 0:
                fov_polygons[frame_idx] = []
                continue

            # Each point has 3 coordinates (x, y, z)
            expected_values = num_points * 3
            if len(parts) < 2 + expected_values:
                continue

            points = []
            for i in range(num_points):
                x = float(parts[2 + i * 3])
                y = float(parts[2 + i * 3 + 1])
                z = float(parts[2 + i * 3 + 2])
                points.append((x, y, z))

            fov_polygons[frame_idx] = points

    return fov_polygons


def draw_fov_polygon(
        map_img: np.ndarray,
        fov_points: List[Tuple[float, float, float]],
        canvas_cfg: dict,
        color: Tuple[int, int, int] = (0, 200, 200),
        thickness: int = 2,
        fill_alpha: float = 0.15
):
    """
    Draw the FOV polygon on the global map.

    Args:
        map_img: The canvas image to draw on
        fov_points: List of (x, y, z) world coordinates
        canvas_cfg: Canvas configuration from make_global_canvas
        color: BGR color for the polygon outline
        thickness: Line thickness for the outline
        fill_alpha: Alpha value for semi-transparent fill (0 = no fill, 1 = solid)
    """
    if not fov_points or len(fov_points) < 3:
        return

    # Convert world coordinates to canvas coordinates
    canvas_points = []
    for x, y, z in fov_points:
        px, py = world_to_canvas(x, y, canvas_cfg)
        canvas_points.append([px, py])

    canvas_points = np.array(canvas_points, dtype=np.int32)

    # Draw semi-transparent fill
    if fill_alpha > 0:
        overlay = map_img.copy()
        cv2.fillPoly(overlay, [canvas_points], color)
        cv2.addWeighted(overlay, fill_alpha, map_img, 1 - fill_alpha, 0, map_img)

    # Draw polygon outline
    cv2.polylines(map_img, [canvas_points], isClosed=True, color=color, thickness=thickness)


# ============================================================
# Map Tile Background
# ============================================================

class MapTileProvider:
    """
    Downloads and caches map tiles from tile servers.
    Supports OpenStreetMap, ESRI, and other compatible tile servers.
    """

    # Common tile server URLs (use {z}, {x}, {y} placeholders)
    OPENSTREETMAP = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    CARTO_LIGHT = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    CARTO_DARK = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
    # Satellite imagery options:
    ESRI_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    GOOGLE_SATELLITE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

    def __init__(self, tile_url: str = None, cache_dir: str = None, utm_epsg: int = 32633):
        """
        Args:
            tile_url: Tile server URL template with {z}, {x}, {y} placeholders
            cache_dir: Directory to cache downloaded tiles (None = no caching)
            utm_epsg: EPSG code for the UTM zone of your coordinates
        """
        self.tile_url = tile_url or self.OPENSTREETMAP
        self.cache_dir = cache_dir
        self.tile_size = 256  # Standard tile size in pixels

        # Transformer from UTM to WGS84 (lat/lon)
        self.transformer = Transformer.from_crs(
            CRS.from_epsg(utm_epsg),
            CRS.from_epsg(4326),
            always_xy=True
        )

        # Request headers to be polite to tile servers
        self.headers = {
            'User-Agent': 'VisualizationScript/1.0 (research purposes)'
        }

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def utm_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert UTM coordinates to latitude/longitude."""
        lon, lat = self.transformer.transform(x, y)
        return lat, lon

    def latlon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates at given zoom level."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x_tile, y_tile

    def tile_to_latlon(self, x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to lat/lon (top-left corner of tile)."""
        n = 2.0 ** zoom
        lon = x_tile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
        lat = math.degrees(lat_rad)
        return lat, lon

    def calculate_zoom_level(self, min_lat: float, max_lat: float,
                             min_lon: float, max_lon: float,
                             canvas_width: int, canvas_height: int) -> int:
        """Calculate appropriate zoom level to fit the extent in the canvas."""
        for zoom in range(18, 0, -1):
            # Get tile range at this zoom
            x1, y1 = self.latlon_to_tile(max_lat, min_lon, zoom)
            x2, y2 = self.latlon_to_tile(min_lat, max_lon, zoom)

            # Calculate pixel dimensions
            tile_span_x = abs(x2 - x1) + 1
            tile_span_y = abs(y2 - y1) + 1
            pixel_width = tile_span_x * self.tile_size
            pixel_height = tile_span_y * self.tile_size

            # Check if it fits reasonably (not too many tiles)
            if tile_span_x <= 10 and tile_span_y <= 10:
                return zoom

        return 1

    def download_tile(self, x: int, y: int, zoom: int) -> Optional[np.ndarray]:
        """Download a single tile and return as numpy array."""
        # Create a hash of the tile URL pattern to separate caches for different sources
        url_hash = hashlib.md5(self.tile_url.encode()).hexdigest()[:8]

        # Check cache first
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{url_hash}_{zoom}_{x}_{y}.png")
            if os.path.exists(cache_path):
                img = cv2.imread(cache_path, cv2.IMREAD_COLOR)
                if img is not None:
                    return img

        # Download tile
        url = self.tile_url.format(z=zoom, x=x, y=y)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                # Decode image
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Cache it
                if self.cache_dir and img is not None:
                    cv2.imwrite(cache_path, img)

                return img
            else:
                print(f"  [WARN] Tile request returned status {response.status_code}: {url}")
        except Exception as e:
            print(f"  [WARN] Failed to download tile {zoom}/{x}/{y}: {e}")

        return None

    def get_map_background(self, global_extent: Tuple[float, float, float, float],
                           canvas_cfg: dict) -> Optional[np.ndarray]:
        """
        Generate a map background image for the given UTM extent.

        Args:
            global_extent: (min_x, max_x, min_y, max_y) in UTM coordinates
            canvas_cfg: Canvas configuration from make_global_canvas

        Returns:
            Background image matching canvas dimensions, or None on failure
        """
        min_x, max_x, min_y, max_y = global_extent

        # Convert UTM corners to lat/lon
        min_lat, min_lon = self.utm_to_latlon(min_x, min_y)
        max_lat, max_lon = self.utm_to_latlon(max_x, max_y)

        # Ensure correct ordering
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon

        canvas_width = canvas_cfg["width"]
        canvas_height = canvas_cfg["height"]
        margin = canvas_cfg["margin"]

        # Calculate zoom level
        zoom = self.calculate_zoom_level(min_lat, max_lat, min_lon, max_lon,
                                         canvas_width, canvas_height)

        # Get tile range
        tile_x1, tile_y1 = self.latlon_to_tile(max_lat, min_lon, zoom)
        tile_x2, tile_y2 = self.latlon_to_tile(min_lat, max_lon, zoom)

        # Ensure correct ordering
        if tile_x1 > tile_x2:
            tile_x1, tile_x2 = tile_x2, tile_x1
        if tile_y1 > tile_y2:
            tile_y1, tile_y2 = tile_y2, tile_y1

        # Download and stitch tiles
        num_tiles_x = tile_x2 - tile_x1 + 1
        num_tiles_y = tile_y2 - tile_y1 + 1

        stitched_width = num_tiles_x * self.tile_size
        stitched_height = num_tiles_y * self.tile_size
        stitched = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

        for ty in range(tile_y1, tile_y2 + 1):
            for tx in range(tile_x1, tile_x2 + 1):
                tile_img = self.download_tile(tx, ty, zoom)
                if tile_img is not None:
                    # Calculate position in stitched image
                    px = (tx - tile_x1) * self.tile_size
                    py = (ty - tile_y1) * self.tile_size
                    stitched[py:py + self.tile_size, px:px + self.tile_size] = tile_img

        # Now we need to crop/transform the stitched image to match our canvas
        # Get the lat/lon bounds of the stitched tiles
        tile_top_lat, tile_left_lon = self.tile_to_latlon(tile_x1, tile_y1, zoom)
        tile_bottom_lat, tile_right_lon = self.tile_to_latlon(tile_x2 + 1, tile_y2 + 1, zoom)

        # Calculate pixel positions of our extent within the stitched image
        def latlon_to_pixel(lat, lon):
            # X position
            px = (lon - tile_left_lon) / (tile_right_lon - tile_left_lon) * stitched_width
            # Y position (latitude decreases as y increases)
            py = (tile_top_lat - lat) / (tile_top_lat - tile_bottom_lat) * stitched_height
            return px, py

        # Get pixel coordinates of our extent corners
        px_min, py_max = latlon_to_pixel(min_lat, min_lon)
        px_max, py_min = latlon_to_pixel(max_lat, max_lon)

        # Extract the region of interest
        px_min, px_max = int(px_min), int(px_max)
        py_min, py_max = int(py_min), int(py_max)

        # Clamp to valid range
        px_min = max(0, px_min)
        py_min = max(0, py_min)
        px_max = min(stitched_width, px_max)
        py_max = min(stitched_height, py_max)

        if px_max <= px_min or py_max <= py_min:
            return None

        cropped = stitched[py_min:py_max, px_min:px_max]

        # Resize to match canvas (excluding margins)
        content_width = canvas_width - 2 * margin
        content_height = canvas_height - 2 * margin

        resized = cv2.resize(cropped, (content_width, content_height))

        # Create final canvas with margins
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[margin:margin + content_height, margin:margin + content_width] = resized

        # Flip vertically because our world_to_canvas flips Y
        canvas[margin:margin + content_height, margin:margin + content_width] = \
            cv2.flip(resized, 0)

        return canvas


def create_map_background(global_extent: Tuple[float, float, float, float],
                          canvas_cfg: dict,
                          utm_epsg: int = 32633,
                          tile_url: str = None,
                          cache_dir: str = None,
                          darken_factor: float = 0.5,
                          utm_offset: Tuple[float, float] = (0.0, 0.0)) -> Optional[np.ndarray]:
    """
    Convenience function to create a map background for the global view.

    Args:
        global_extent: (min_x, max_x, min_y, max_y) in local/UTM coordinates
        canvas_cfg: Canvas configuration from make_global_canvas
        utm_epsg: EPSG code for UTM zone (default: 32633 for UTM zone 33N)
        tile_url: Custom tile server URL (default: OpenStreetMap)
        cache_dir: Directory to cache tiles (default: None)
        darken_factor: Darken the map for better overlay visibility (0-1, lower = darker)
        utm_offset: (x_offset, y_offset) to convert local coords to absolute UTM

    Returns:
        Background image or None on failure
    """
    # Convert local extent to absolute UTM by adding offsets
    min_x, max_x, min_y, max_y = global_extent
    abs_extent = (
        min_x + utm_offset[0],
        max_x + utm_offset[0],
        min_y + utm_offset[1],
        max_y + utm_offset[1]
    )

    print(f"  Local extent: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
    print(
        f"  Absolute UTM extent: X=[{abs_extent[0]:.1f}, {abs_extent[1]:.1f}], Y=[{abs_extent[2]:.1f}, {abs_extent[3]:.1f}]")

    provider = MapTileProvider(tile_url=tile_url, cache_dir=cache_dir, utm_epsg=utm_epsg)
    background = provider.get_map_background(abs_extent, canvas_cfg)

    if background is not None and darken_factor < 1.0:
        background = (background * darken_factor).astype(np.uint8)

    return background


# ============================================================
# MAIN: live visualization over multiple sequences
# ============================================================

if __name__ == "__main__":
    georef_dets_base_folder = r"Z:\dets\georeferenced5"  # r"Z:\dets\georeferenced_smoothed"
    detections_base_folder = r"Z:\dets\source"
    images_base_folder = r"Z:\sequences"

    iou_thresh = 0.3
    class_aware = True
    max_age = -1
    minimum_confidence = 0.3
    tracker = gt.TrackerMode.HUNGARIAN
    max_center_distance = 0.2  # only used with TrackerMode.CENTER or TrackerMode.HUNGARIAN_CENTER

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

    # NEW: FOV polygon visualization options
    show_fov_polygon = True  # Set to True to display the georeferenced FOV polygon
    fov_data_folder = r"Z:\dets\georeferenced_fov"  # Folder containing {parent}_fov_georeferenced.txt files
    fov_polygon_color = (0, 200, 200)  # BGR color for FOV polygon (cyan/teal)
    fov_polygon_thickness = 2  # Line thickness
    fov_polygon_fill_alpha = 0.15  # Semi-transparent fill (0 = no fill)

    # NEW: Map tile background options
    show_map_background = True  # Set to True to display map tiles as background
    map_tile_cache_dir = r"Z:\map_tile_cache"  # Directory to cache downloaded tiles (None = no caching)
    map_utm_epsg = 32633  # EPSG code for your UTM zone (32633 = UTM zone 33N)
    map_darken_factor = 0.4  # Darken map for better overlay visibility (0-1, lower = darker)
    # Tile server options - uncomment the one you want to use:
    # map_tile_url = None  # Default: OpenStreetMap
    map_tile_url = MapTileProvider.OPENSTREETMAP  # Standard map view
    # map_tile_url = MapTileProvider.CARTO_DARK  # Dark themed map (good for bright overlays)
    # map_tile_url = MapTileProvider.CARTO_LIGHT  # Light themed map
    # map_tile_url = MapTileProvider.ESRI_SATELLITE  # Satellite imagery (free, no API key)
    # map_tile_url = MapTileProvider.GOOGLE_SATELLITE  # Google satellite (may violate ToS)

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
        "polyorder": polyorder,
        "show_fov_polygon": show_fov_polygon,
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

            if not file.startswith("14_1"):
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

        # NEW: Load FOV polygons if enabled
        fov_polygons = {}
        if show_fov_polygon:
            fov_file_path = os.path.join(fov_data_folder, f"{flight_id}_fov_georeferenced.txt")
            if os.path.exists(fov_file_path):
                fov_polygons = load_fov_polygons(fov_file_path)
                print(f"  Loaded FOV polygons for {len(fov_polygons)} frames")
            else:
                print(f"  [WARN] FOV file not found: {fov_file_path}")

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

        # NEW: Load map background if enabled
        map_background = None
        if show_map_background:
            print("  Loading map tiles...")
            map_background = create_map_background(
                global_extent,
                canvas_cfg,
                utm_epsg=map_utm_epsg,
                tile_url=map_tile_url,
                cache_dir=map_tile_cache_dir,
                darken_factor=map_darken_factor,
                utm_offset=(0, 0)
            )
            if map_background is not None:
                print("  Map background loaded successfully")
            else:
                print("  [WARN] Could not load map background")

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
                    interpolated = tinfo["interpolated"]

                    if row_idx in pixel_bboxes_by_row:
                        x1, y1, x2, y2, det_conf, cls = pixel_bboxes_by_row[row_idx]

                        if conf >= 0 and interpolated == 0:
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
            if map_background is not None:
                map_img = map_background.copy()
            else:
                map_img = np.zeros(
                    (canvas_cfg["height"], canvas_cfg["width"], 3), dtype=np.uint8
                )

            # NEW: Draw FOV polygon first (so it's behind other elements)
            if show_fov_polygon and fov_polygons:
                try:
                    frame_idx = int(frame_id)
                    if frame_idx in fov_polygons:
                        fov_points = fov_polygons[frame_idx]
                        draw_fov_polygon(
                            map_img,
                            fov_points,
                            canvas_cfg,
                            color=fov_polygon_color,
                            thickness=fov_polygon_thickness,
                            fill_alpha=fov_polygon_fill_alpha
                        )
                except ValueError:
                    pass

            # Current frame's track boxes + track history
            if frame_id in sequence_tracks:
                for tinfo in sequence_tracks[frame_id]:
                    tid = tinfo["tid"]
                    color = id_to_color(tid)
                    conf = tinfo["conf"]
                    interpolated = tinfo["interpolated"]

                    gx1, gy1 = tinfo["gx1"], tinfo["gy1"]
                    gx2, gy2 = tinfo["gx2"], tinfo["gy2"]

                    px1, py1 = world_to_canvas(gx1, gy1, canvas_cfg)
                    px2, py2 = world_to_canvas(gx2, gy2, canvas_cfg)

                    if conf >= 0 and interpolated == 0:
                        cv2.rectangle(map_img, (px1, py1), (px2, py2), color, 2)
                    else:
                        draw_dashed_rectangle(
                            map_img,
                            (px1, py1),
                            (px2, py2),
                            color,
                            2,
                            8,  # shorter dash length for smaller global view boxes
                        )

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