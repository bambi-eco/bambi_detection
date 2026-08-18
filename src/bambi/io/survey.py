# -*- coding: utf-8 -*-
"""The file edge for the survey-analytics files the pipeline and the QGIS
plugin exchange: flight-route GeoJSON, ``fov_polygons.txt``, the
perpendicular-distance JSONs, transects JSON/CSV. Readers return arrays;
writers reproduce the plugin's layouts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

PathLike = Union[str, Path]

__all__ = ["read_route_geojson", "write_route_geojson", "write_points_geojson", "read_fov_polygons",
           "write_fov_polygons", "write_perpendicular_json", "read_perpendicular_json",
           "read_transects", "write_transects"]


def _crs_block(epsg: int) -> dict:
    return {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{int(epsg)}"}}


# ---------------------------------------------------------------- flight route

def read_route_geojson(path: PathLike) -> NDArray[np.float64]:
    """The first LineString (or Point) of a GeoJSON as ``(M, 2|3)`` coordinates."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") == "LineString":
            return np.asarray(geom["coordinates"], dtype=np.float64)
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Point":
            return np.asarray([geom["coordinates"]], dtype=np.float64)
    raise ValueError(f"no LineString/Point feature in {path}")


def write_route_geojson(path: PathLike, coordinates: ArrayLike, epsg: int, name: str = "flight_route") -> Path:
    """The plugin's ``flight_route.geojson``: one LineString feature (a Point for a single fix)."""
    c = np.asarray(coordinates, dtype=np.float64)
    coords = c.tolist()
    geom = {"type": "Point", "coordinates": coords[0]} if len(coords) == 1 else {"type": "LineString",
                                                                                 "coordinates": coords}
    data = {"type": "FeatureCollection", "name": name, "crs": _crs_block(epsg),
            "features": [{"type": "Feature", "geometry": geom,
                          "properties": {"name": "Flight Route", "total_gps_points": len(coords)}}]}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def write_points_geojson(path: PathLike, points: ArrayLike, epsg: int, properties: Optional[List[dict]] = None,
                         name: str = "camera_positions") -> Path:
    """A FeatureCollection of Point features (``(N, 2|3)``) with optional per-point properties."""
    pts = np.asarray(points, dtype=np.float64).tolist()
    props = properties or [{} for _ in pts]
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": xy}, "properties": pr}
             for xy, pr in zip(pts, props)]
    data = {"type": "FeatureCollection", "name": name, "crs": _crs_block(epsg), "features": feats}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------- fov_polygons.txt

def read_fov_polygons(path: PathLike) -> Dict[int, NDArray[np.float64]]:
    """``# frame_idx num_points x1 y1 z1 ...`` -> ``{frame: (K, 3)}`` (frames with 0 points map to ``(0, 3)``)."""
    out: Dict[int, NDArray[np.float64]] = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        frame, n = int(parts[0]), int(parts[1])
        vals = np.asarray(parts[2:2 + 3 * n], dtype=np.float64)
        out[frame] = vals.reshape(-1, 3) if n else np.zeros((0, 3))
    return out


def write_fov_polygons(path: PathLike, polygons: Dict[int, ArrayLike]) -> Path:
    """The plugin's ``fov_polygons.txt`` (world coordinates, 6 decimals)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("# FoV polygon georeferenced data\n")
        f.write("# Format: frame_idx num_points x1 y1 z1 x2 y2 z2 ...\n")
        for frame in sorted(polygons):
            q = np.asarray(polygons[frame], dtype=np.float64).reshape(-1, 3) if len(polygons[frame]) else np.zeros((0, 3))
            q = q[np.isfinite(q).all(axis=1)]
            if len(q) == 0:
                f.write(f"{int(frame)} 0\n")
            else:
                f.write(f"{int(frame)} {len(q)} " + " ".join(f"{v:.6f}" for v in q.ravel()) + "\n")
    return p


# ---------------------------------------------------------------- perpendicular JSON

def write_perpendicular_json(path: PathLike, frames: ArrayLike, centres: ArrayLike, feet: ArrayLike,
                             distances: ArrayLike, confidences: ArrayLike, classes: ArrayLike, epsg: int,
                             ids: Optional[ArrayLike] = None, kind: str = "detections") -> Path:
    """The plugin's ``perpendicular_{m}.json`` (``kind="detections"``, list key
    ``perpendiculas``) or ``perpendicular_tracks_{m}.json`` (``kind="tracks"``)."""
    fr = np.asarray(frames).astype(int)
    c = np.asarray(centres, dtype=np.float64).reshape(len(fr), -1)
    ft = np.asarray(feet, dtype=np.float64).reshape(len(fr), -1)
    d = np.asarray(distances, dtype=np.float64)
    conf = np.asarray(confidences, dtype=np.float64)
    cls = np.asarray(classes).astype(int)
    idv = np.arange(len(fr)) if ids is None else np.asarray(ids)
    rows = []
    for i in range(len(fr)):
        cz = float(c[i, 2]) if c.shape[1] > 2 else 0.0
        row = {"frame": int(fr[i]), "confidence": float(conf[i]), "class_id": int(cls[i]),
               "detection_center": [float(c[i, 0]), float(c[i, 1]), cz],
               "foot_point": [float(ft[i, 0]), float(ft[i, 1])], "distance_m": round(float(d[i]), 4)}
        if kind == "tracks":
            row = {"track_id": int(idv[i]), "last_frame": int(fr[i]), **{k: v for k, v in row.items() if k != "frame"}}
        else:
            row = {"det_idx": int(idv[i]), **row}
        rows.append(row)
    key = "tracks" if kind == "tracks" else "perpendiculas"
    data = {"crs": f"EPSG:{int(epsg)}", f"total_{kind}": len(rows), key: rows}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def read_perpendicular_json(path: PathLike) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64],
                                                    NDArray[np.float64], NDArray[np.int64]]:
    """Either perpendicular JSON -> ``(frames, centres (N,3), feet (N,2), distances, classes)``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("perpendiculas") or data.get("tracks") or []
    frames = np.array([r.get("frame", r.get("last_frame", -1)) for r in rows], dtype=np.int64)
    centres = np.array([r["detection_center"] for r in rows], dtype=np.float64).reshape(-1, 3)
    feet = np.array([r["foot_point"] for r in rows], dtype=np.float64).reshape(-1, 2)
    dist = np.array([r["distance_m"] for r in rows], dtype=np.float64)
    cls = np.array([r.get("class_id", 0) or 0 for r in rows], dtype=np.int64)
    return frames, centres, feet, dist, cls


# ---------------------------------------------------------------- transects

def read_transects(path: PathLike) -> Tuple[NDArray[np.int64], NDArray[np.int64], List[str]]:
    """``transects.json`` -> ``(ids (K,), ranges (K, 2) [start, end], names)``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("transects", [])
    ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)
    ranges = np.array([[int(r.get("start_frame", 0)), int(r.get("end_frame", 0))] for r in rows],
                      dtype=np.int64).reshape(-1, 2)
    return ids, ranges, [str(r.get("name", "") or "") for r in rows]


def write_transects(path: PathLike, ranges: ArrayLike, modality: str = "t", names: Optional[Sequence[str]] = None,
                    ids: Optional[ArrayLike] = None, lengths_m: Optional[ArrayLike] = None,
                    timestamps: Optional[Sequence[str]] = None) -> Path:
    """The plugin's ``transects.json`` (+ ``transects.csv`` beside it)."""
    r = np.asarray(ranges).astype(int).reshape(-1, 2)
    k = len(r)
    idv = (np.arange(1, k + 1) if ids is None else np.asarray(ids)).astype(int)
    nm = list(names) if names is not None else [""] * k
    ln = None if lengths_m is None else np.asarray(lengths_m, dtype=np.float64)

    def ts(frame):
        return str(timestamps[frame]) if timestamps is not None and 0 <= frame < len(timestamps) else ""

    rows = [{"id": int(idv[i]), "name": nm[i], "start_frame": int(r[i, 0]), "end_frame": int(r[i, 1]),
             "start_time": ts(min(r[i])), "end_time": ts(max(r[i])),
             "length_m": None if ln is None else round(float(ln[i]), 2)} for i in range(k)]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 1, "modality": modality, "transects": rows}, indent=2), encoding="utf-8")
    with open(p.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "start_frame", "end_frame", "start_time", "end_time", "length_m"])
        for e in rows:
            w.writerow([e["id"], e["name"] or "", e["start_frame"], e["end_frame"], e["start_time"], e["end_time"],
                        "" if e["length_m"] is None else e["length_m"]])
    return p
