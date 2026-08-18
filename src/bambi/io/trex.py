# -*- coding: utf-8 -*-
"""The file edge for TRex tracklets: ``*.npz`` -> per-frame boxes.

`TRex <https://trex.run>`_ writes one ``.npz`` per tracked individual with
per-frame pose key-points (``poseX0..N`` / ``poseY0..N``), a detection
probability and class, and the raw video size. This reads a folder of them
into arrays: one row per (frame, individual) with the axis-aligned box
around the finite key-points - which is exactly how the QGIS plugin's
"Import TRex Tracklets" turns them into detections.

Boxes are in *raw video* pixels. To put them on the extracted frames (and
so onto the DEM) run them through
:func:`bambi.geo.calibration.undistort_boxes` with the calibration the
frames were extracted with.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, Path]

__all__ = ["TrexTracklets", "read_trex_tracklets", "read_trex_file"]


@dataclass(frozen=True)
class TrexTracklets:
    """All individuals of a folder, one row per (frame, individual), sorted by ``(frame, track_id)``."""
    frames: NDArray[np.int64]
    track_ids: NDArray[np.int64]
    boxes: NDArray[np.float64]           # (N, 4) xyxy raw-video pixels
    confidences: NDArray[np.float64]
    classes: NDArray[np.int64]
    video_size: Optional[Tuple[int, int]]   # (width, height) of the raw video, when the files carry it
    files: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.frames)


def _track_id(data, path: Path, fallback: int) -> int:
    if "id" in data and len(np.asarray(data["id"]).ravel()):
        return int(np.asarray(data["id"]).ravel()[0])
    stem = path.stem
    return int(stem.split("id")[-1]) if "id" in stem and stem.split("id")[-1].isdigit() else fallback


def read_trex_file(path: PathLike, fallback_id: int = 0):
    """One ``.npz`` -> ``(frames, track_id, boxes (N, 4), confidences, classes, video_size)``.

    Rows without any finite key-point are dropped (TRex writes NaN for
    frames where the individual was not seen).
    """
    p = Path(path)
    data = np.load(str(p), allow_pickle=True)
    video_size = None
    if "video_size" in data:
        vs = np.asarray(data["video_size"]).ravel()
        if len(vs) >= 2 and vs[0] > 0 and vs[1] > 0:
            video_size = (int(vs[0]), int(vs[1]))
    tid = _track_id(data, p, fallback_id)
    raw_frames = np.asarray(data["frame"], dtype=np.float64).ravel()
    n = len(raw_frames)
    frames = np.where(np.isfinite(raw_frames), raw_frames, -1).astype(np.int64)
    if "detection_p" in data:
        raw_conf = np.asarray(data["detection_p"], dtype=np.float64).ravel()
        conf = np.where(np.isfinite(raw_conf), raw_conf, 0.0)
    else:
        conf = np.ones(n)
    if "detection_class" in data:
        raw_cls = np.asarray(data["detection_class"], dtype=np.float64).ravel()
        cls = np.where(np.isfinite(raw_cls), raw_cls, 0.0).astype(np.int64)
    else:
        cls = np.zeros(n, np.int64)
    keys = sorted(int(k[len("poseX"):]) for k in data.keys()
                  if k.startswith("poseX") and k[len("poseX"):].isdigit() and f"poseY{k[len('poseX'):]}" in data)
    if not keys:
        empty = np.zeros(0)
        return empty.astype(np.int64), tid, np.zeros((0, 4)), empty, empty.astype(np.int64), video_size
    xs = np.stack([np.asarray(data[f"poseX{k}"], dtype=np.float64).ravel() for k in keys], axis=1)   # (n, K)
    ys = np.stack([np.asarray(data[f"poseY{k}"], dtype=np.float64).ravel() for k in keys], axis=1)
    ok = np.isfinite(xs) & np.isfinite(ys)
    keep = ok.any(axis=1) & (frames >= 0)
    x1 = np.where(ok, xs, np.inf).min(axis=1)
    y1 = np.where(ok, ys, np.inf).min(axis=1)
    x2 = np.where(ok, xs, -np.inf).max(axis=1)
    y2 = np.where(ok, ys, -np.inf).max(axis=1)
    boxes = np.column_stack([x1, y1, x2, y2])[keep]
    conf = conf[:n][keep] if len(conf) >= n else np.pad(conf, (0, n - len(conf)), constant_values=1.0)[keep]
    cls = cls[:n][keep] if len(cls) >= n else np.pad(cls, (0, n - len(cls)))[keep]
    return frames[keep], tid, boxes, conf, cls, video_size


def read_trex_tracklets(folder: PathLike, pattern: str = "*.npz") -> TrexTracklets:
    """Every tracklet file in ``folder``, merged and sorted by ``(frame, track_id)``."""
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {folder}")
    F, T, B, C, K = [], [], [], [], []
    video_size = None
    for i, f in enumerate(files):
        fr, tid, boxes, conf, cls, vs = read_trex_file(f, fallback_id=i)
        if video_size is None:
            video_size = vs
        F.append(fr)
        T.append(np.full(len(fr), tid, np.int64))
        B.append(boxes)
        C.append(conf)
        K.append(cls)
    frames = np.concatenate(F)
    tids = np.concatenate(T)
    o = np.lexsort((tids, frames))
    return TrexTracklets(frames[o], tids[o], np.vstack(B)[o], np.concatenate(C)[o], np.concatenate(K)[o],
                         video_size, tuple(str(f) for f in files))
