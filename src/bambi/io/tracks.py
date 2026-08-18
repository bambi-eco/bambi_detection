# -*- coding: utf-8 -*-
"""The file edge for detections and tracks: the pipeline's text tables as arrays.

Five plain-text tables travel through the pipeline and the QGIS plugin; each
has one reader and one writer here, and the writers reproduce the plugin's
formatting to the byte so a file written by the engine is a file the plugin
would have written:

============================  =====================================================================
``detections.txt``            ``# frame x1 y1 x2 y2 confidence class_id`` (space-separated)
``georeferenced.txt``         ``# idx frame min_x min_y min_z max_x max_y max_z confidence class_id``
``tracks.csv``                ``frame(08d),track_id,x1,y1,z1,x2,y2,z2,confidence,class_id,interpolated``
``tracks_pixel.csv``          ``frame,track_id,x1,y1,x2,y2,conf,cls,interpolated`` (header optional)
MOT (``*_gt.txt``)            ``frame,track_id,left,top,width,height,confidence,class,visibility[,...]``
============================  =====================================================================

Everything comes back as a :class:`Table` of parallel arrays; the compute
modules (:mod:`bambi.tracking`, :mod:`bambi.geo.georef`) never see a path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bambi.tracking.iou import Tracks

PathLike = Union[str, Path]

__all__ = ["Table", "read_detections", "write_detections", "read_georeferenced", "write_georeferenced",
           "read_tracks_csv", "write_tracks_csv", "read_pixel_tracks", "write_pixel_tracks",
           "read_mot", "write_mot", "tracks_from_table"]


@dataclass(frozen=True)
class Table:
    """Detections or track rows as parallel arrays, in file order.

    ``track_ids`` is all ``-1`` for tables that carry no track; ``boxes`` is
    ``(N, 4)`` xyxy for pixel tables and ``(N, 6)`` ``[x1 y1 z1 x2 y2 z2]``
    for geo-referenced ones. ``extra`` holds any trailing columns as strings.
    """
    frames: NDArray[np.int64]
    boxes: NDArray[np.float64]
    confidences: NDArray[np.float64]
    classes: NDArray[np.int64]
    track_ids: NDArray[np.int64]
    interpolated: NDArray[np.bool_]
    extra: Optional[NDArray] = None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def is_geo(self) -> bool:
        return self.boxes.shape[1] == 6


def _rows(path: PathLike, sep: Optional[str]):
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cells = s.split(sep) if sep else s.split()
        if not cells or not cells[0].strip().lstrip("-").isdigit():
            continue                       # header row without '#'
        out.append([c.strip() for c in cells])
    return out


def _table(rows, ncols_min, frame_col, box_cols, conf_col, cls_col, tid_col=None, interp_col=None) -> Table:
    n = len(rows)
    if n == 0:
        return Table(np.zeros(0, np.int64), np.zeros((0, len(box_cols))), np.zeros(0), np.zeros(0, np.int64),
                     np.full(0, -1, np.int64), np.zeros(0, bool))
    width = min(len(r) for r in rows)
    if width < ncols_min:
        raise ValueError(f"expected at least {ncols_min} columns, got {width}")
    frames = np.array([int(float(r[frame_col])) for r in rows], np.int64)
    boxes = np.array([[float(r[c]) for c in box_cols] for r in rows], np.float64)
    conf = np.array([float(r[conf_col]) if conf_col is not None and conf_col < len(r) else 1.0 for r in rows])
    cls = np.array([int(float(r[cls_col])) if cls_col is not None and cls_col < len(r) else 0 for r in rows], np.int64)
    tid = (np.array([int(float(r[tid_col])) for r in rows], np.int64) if tid_col is not None
           else np.full(n, -1, np.int64))
    interp = (np.array([bool(int(float(r[interp_col]))) if interp_col < len(r) else False for r in rows])
              if interp_col is not None else np.zeros(n, bool))
    used = max([frame_col, conf_col or 0, cls_col or 0, tid_col or 0, interp_col or 0] + list(box_cols)) + 1
    extra = None
    if any(len(r) > used for r in rows):
        extra = np.array([r[used:] + [""] * (max(len(x) for x in rows) - len(r)) for r in rows], dtype=object)
    return Table(frames, boxes, conf, cls, tid, interp, extra)


# ---------------------------------------------------------------- detections.txt

def read_detections(path: PathLike) -> Table:
    """``# frame x1 y1 x2 y2 confidence class_id`` -> pixel boxes."""
    return _table(_rows(path, None), 5, 0, (1, 2, 3, 4), 5, 6)


def write_detections(path: PathLike, frames: ArrayLike, boxes: ArrayLike, confidences: ArrayLike,
                     classes: ArrayLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("# frame x1 y1 x2 y2 confidence class_id\n")
        for fr, b, c, k in zip(np.asarray(frames), np.asarray(boxes, float), np.asarray(confidences, float),
                               np.asarray(classes)):
            f.write(f"{int(fr)} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} {c:.4f} {int(k)}\n")
    return p


# ---------------------------------------------------------------- georeferenced.txt

def read_georeferenced(path: PathLike) -> Table:
    """``# idx frame min_x min_y min_z max_x max_y max_z confidence class_id`` -> geo boxes."""
    return _table(_rows(path, None), 8, 1, (2, 3, 4, 5, 6, 7), 8, 9)


def write_georeferenced(path: PathLike, frames: ArrayLike, boxes: ArrayLike, confidences: ArrayLike,
                        classes: ArrayLike) -> Path:
    """Write ``(N, 6)`` ``[x1 y1 z1 x2 y2 z2]`` rows the way the plugin does."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("# idx frame min_x min_y min_z max_x max_y max_z confidence class_id\n")
        for i, (fr, b, c, k) in enumerate(zip(np.asarray(frames), np.asarray(boxes, float),
                                              np.asarray(confidences, float), np.asarray(classes))):
            f.write(f"{i} {int(fr)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.6f} "
                    f"{c:.4f} {int(k)}\n")
    return p


# ---------------------------------------------------------------- tracks.csv

def read_tracks_csv(path: PathLike) -> Table:
    """``frame,track_id,x1,y1,z1,x2,y2,z2,confidence,class_id[,interpolated]``."""
    return _table(_rows(path, ","), 10, 0, (2, 3, 4, 5, 6, 7), 8, 9, tid_col=1, interp_col=10)


def write_tracks_csv(path: PathLike, tracks: Tracks) -> Path:
    """Geo-referenced tracks in the plugin's ``tracks.csv`` format (boxes must be ``(N, 6)``)."""
    if tracks.boxes.shape[1] != 6:
        raise ValueError("tracks.csv carries geo boxes: (N, 6) [x1 y1 z1 x2 y2 z2]")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        for fr, t, b, c, k, i in zip(tracks.frames, tracks.track_ids, tracks.boxes, tracks.confidences,
                                     tracks.classes, tracks.interpolated):
            f.write(f"{int(fr):08d},{int(t)},{b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f},{b[4]:.6f},{b[5]:.6f},"
                    f"{c:.6f},{int(k)},{int(i)}\n")
    return p


# ---------------------------------------------------------------- tracks_pixel.csv

def read_pixel_tracks(path: PathLike) -> Table:
    """``frame,track_id,x1,y1,x2,y2,conf,cls,interpolated`` (header optional, ``08d`` frames tolerated)."""
    return _table(_rows(path, ","), 8, 0, (2, 3, 4, 5), 6, 7, tid_col=1, interp_col=8)


def write_pixel_tracks(path: PathLike, tracks: Tracks, style: str = "plugin") -> Path:
    """Pixel tracks; ``style="plugin"`` (header, 2 decimals) or ``"trex"`` (no header, 08d, 6 decimals)."""
    if tracks.boxes.shape[1] != 4:
        raise ValueError("tracks_pixel.csv carries pixel boxes: (N, 4) [x1 y1 x2 y2]")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        if style == "plugin":
            f.write("# frame,track_id,x1,y1,x2,y2,conf,cls,interpolated\n")
        for fr, t, b, c, k, i in zip(tracks.frames, tracks.track_ids, tracks.boxes, tracks.confidences,
                                     tracks.classes, tracks.interpolated):
            if style == "plugin":
                f.write(f"{int(fr)},{int(t)},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},{c:.4f},{int(k)},{int(i)}\n")
            else:
                f.write(f"{int(fr):08d},{int(t)},{b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f},{c:.6f},{int(k)},{int(i)}\n")
    return p


# ---------------------------------------------------------------- MOT

def read_mot(path: PathLike) -> Table:
    """MOT challenge rows -> xyxy pixel boxes; trailing columns land in ``extra``."""
    rows = _rows(path, ",")
    t = _table(rows, 6, 0, (2, 3, 4, 5), 6 if rows and len(rows[0]) > 6 else None,
               7 if rows and len(rows[0]) > 7 else None, tid_col=1)
    boxes = t.boxes.copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    used = 8 if rows and len(rows[0]) > 7 else (7 if rows and len(rows[0]) > 6 else 6)
    extra = None
    if rows and any(len(r) > used for r in rows):
        width = max(len(r) for r in rows)
        extra = np.array([r[used:] + [""] * (width - len(r)) for r in rows], dtype=object)
    return Table(t.frames, boxes, t.confidences, t.classes, t.track_ids, t.interpolated, extra)


def write_mot(path: PathLike, tracks: Tracks, visibility: Optional[ArrayLike] = None) -> Path:
    """``frame,track_id,left,top,width,height,confidence,class,visibility`` from ``(N, 4)`` xyxy tracks."""
    if tracks.boxes.shape[1] != 4:
        raise ValueError("MOT carries pixel boxes: (N, 4) [x1 y1 x2 y2]")
    vis = np.ones(len(tracks)) if visibility is None else np.asarray(visibility, float)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        for fr, t, b, c, k, v in zip(tracks.frames, tracks.track_ids, tracks.boxes, tracks.confidences,
                                     tracks.classes, vis):
            f.write(f"{int(fr)},{int(t)},{b[0]:.2f},{b[1]:.2f},{b[2] - b[0]:.2f},{b[3] - b[1]:.2f},"
                    f"{c:.2f},{int(k)},{v:.2f}\n")
    return p


# ---------------------------------------------------------------- glue

def tracks_from_table(table: Table) -> Tracks:
    """A :class:`Table` that carries track ids -> :class:`bambi.tracking.iou.Tracks` (sorted)."""
    if (table.track_ids < 0).any():
        raise ValueError("table has rows without a track id")
    o = np.lexsort((table.track_ids, table.frames))
    return Tracks(table.frames[o], table.track_ids[o], table.boxes[o], table.confidences[o], table.classes[o],
                  table.interpolated[o], o.astype(np.int64))
