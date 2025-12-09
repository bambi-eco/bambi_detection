import colorsys
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict

from pyproj import Transformer, CRS

from src.bambi.extend_geo_referenced_tracks import read_tracks_csv
from src.bambi.georeference_deepsort_mot import deviating_folders


def center_of_bbox(xmin: float, ymin: float, xmax: float, ymax: float) -> Tuple[float, float]:
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    return cx, cy

def bbox_ring_utm(x1: float, y1: float, x2: float, y2: float):
    """Return a closed ring [(x,y), ...] in UTM for the axis-aligned bbox."""
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),  # close ring
    ]

def id_to_color(identifier, *, saturation=0.65, lightness=0.5):
    """
    Deterministically map any identifier (int/str/etc.) to a distinct BGR color for OpenCV.
    """
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2**32  # [0,1)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)  # RGB in [0,1]
    # OpenCV uses BGR and expects ints
    return (int(b * 255), int(g * 255), int(r * 255))

def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """Convert BGR ints (0..255) to #RRGGBB hex."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

if __name__ == '__main__':
    tracks_base = r"Z:\dets\georeferenced"
    target_base = r"Z:\dets\georeferenced\geojson"
    rel_transformer = Transformer.from_crs(CRS.from_epsg(32633), CRS.from_epsg(4326))
    transform_to_target_crs = False
    export_single_tracks = False
    export_complete_flight = True

    #############################################

    parent_dict = defaultdict(list)

    for root, dirs, files in os.walk(tracks_base):
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
        for f in files:
            p = Path(f)
            target_folder = os.path.join(target_base, deviating_folders(tracks_base, f))
            os.makedirs(target_folder, exist_ok=True)
            tracks = read_tracks_csv(f)
            converted_tracks: Dict[str, List[Tuple[int, float, float, float]]] = defaultdict(list)
            points_meta: Dict[str, List[dict]] = defaultdict(list)
            detections = defaultdict(dict)
            for track_id, track in tracks.items():
                for (frame_id, det) in track:
                    cx, cy = center_of_bbox(det.x1, det.y1, det.x2, det.y2)
                    lat, lon = rel_transformer.transform(cx, cy)
                    converted_tracks[track_id].append((frame_id, lon, lat, det.conf))
                    points_meta[track_id].append({
                        "frame": frame_id,
                        "score": det.conf,
                        "utm_center_east": cx,
                        "utm_center_north": cy
                    })

                    lat1, lon1 = rel_transformer.transform(det.x1, det.y1)
                    lat2, lon2 = rel_transformer.transform(det.x2, det.y2)
                    detections[track_id][frame_id] = [lon1, lat1, lon2, lat2]

            exported = 0
            combined_features = []
            for tid, items in converted_tracks.items():
                # sort by frame
                items.sort(key=lambda t: t[0])

                track_color_hex = bgr_to_hex(id_to_color(tid))

                coords = [[lon, lat] for (_f, lon, lat, _s) in items]
                if len(coords) == 1:
                    coords = [coords[0], coords[0]]
                frames = [f for (f, _lon, _lat, _s) in items]
                scores = [s for (_f, _lon, _lat, s) in items]

                # Build GeoJSON with one LineString + Points (each detection as a Feature)
                features = []

                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "track_id": tid,
                        "frames": frames,
                        "scores": scores,
                        # "crs_input": f"EPSG:{args.utm_epsg}",
                        # "crs_output": f"EPSG:{args.wgs84_epsg}",
                        "note": "Coordinates are [lon, lat] in WGS84.",
                        "stroke": track_color_hex,
                        "stroke-width": 2,
                        "stroke-opacity": 1.0,
                    }
                })

                for (frame_id, det) in detections[tid].items():
                    # 1) build UTM ring
                    x1 = det[0]
                    y1 = det[1]
                    x2 = det[2]
                    y2 = det[3]
                    ring_lonlat = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]

                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [ring_lonlat]},
                        "properties": {
                            "track_id": tid,
                            "frame": frame_id,
                            "note": "Polygon is bbox corners transformed to WGS84 [lon, lat].",
                            "stroke": track_color_hex,
                            "stroke-width": 1,
                            "stroke-opacity": 0.9,
                            "fill": track_color_hex,
                            "fill-opacity": 0.25,
                        }
                    })

                # Optional per-detection points
                for (frame, lon, lat, score), meta in zip(items, points_meta[tid]):
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {
                            "track_id": tid,
                            "frame": frame,
                            "score": score,
                            # include original UTM center for traceability
                            "utm_center_east": meta["utm_center_east"],
                            "utm_center_north": meta["utm_center_north"],
                            # "crs_input": f"EPSG:{args.utm_epsg}",
                            "marker-color": track_color_hex,
                            "marker-size": "small",
                        }
                    })

                fc = {"type": "FeatureCollection", "features": features}

                target_file = os.path.join(target_folder, p.stem + f"_{tid}.json")
                if export_single_tracks:
                    with open(target_file, "w", encoding="utf-8") as f:
                        json.dump(fc, f, ensure_ascii=False, indent=2)
                    print(f"Exported {target_file}")

                combined_features.extend(features)

            combined_fc = {
                "type": "FeatureCollection",
                "features": combined_features,
            }
            combined_path = os.path.join(target_folder, p.stem + ".json")
            if export_complete_flight:
                with open(combined_path, "w", encoding="utf-8") as f:
                    json.dump(combined_fc, f, ensure_ascii=False, indent=2)
                print(f"Exported {combined_path}")
