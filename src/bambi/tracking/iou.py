# -*- coding: utf-8 -*-
"""The built-in tracker, on arrays.

Frame by frame, detections are matched to the live tracks by box overlap
(IoU) - greedily or with the Hungarian assignment - optionally with a
centre-distance fallback for what overlap could not pair. Unmatched
detections open new tracks; tracks that go unmatched age out after
``max_age`` frames. It is the algorithm the QGIS plugin's "built-in" backend
and ``bambi.georeferenced_tracking`` run, re-stated so that

* input is ``(N,)`` frames + ``(N, D)`` boxes (+ classes), output is an
  ``(N,)`` array of track ids aligned with the input - nothing is reordered
  or dropped;
* boxes may be pixel ``[x1, y1, x2, y2]`` or DEM-local
  ``[x1, y1, z1, x2, y2, z2]`` (overlap is computed on the horizontal
  extent either way, exactly as before);
* the per-frame maths is vectorised, while tie-breaking is kept identical
  to the scalar original (float32 IoU, stable sorts), so results are the
  same rows, in the same order, with the same ids.

:func:`interpolate_tracks` then fills frame gaps inside every track with
linearly interpolated boxes, marking them, and keeps an index back to the
detection each output row came from (``-1`` for interpolated rows).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MODES", "iou_matrix", "track_boxes", "interpolate_tracks", "Tracks", "split_boxes"]

MODES = ("greedy", "hungarian", "center", "hungarian_center")
_LARGE = 1e6


def split_boxes(boxes: ArrayLike) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """``(N, 4)`` -> ``(xyxy, None)``; ``(N, 6)`` ``[x1 y1 z1 x2 y2 z2]`` -> ``(xyxy, z (N, 2))``."""
    b = np.asarray(boxes, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] not in (4, 6):
        raise ValueError(f"boxes must be (N, 4) [x1 y1 x2 y2] or (N, 6) [x1 y1 z1 x2 y2 z2], got {b.shape}")
    if b.shape[1] == 4:
        return b, None
    return b[:, [0, 1, 3, 4]], b[:, [2, 5]]


def iou_matrix(a: ArrayLike, b: ArrayLike) -> NDArray[np.float32]:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)`` float32.

    float32 on purpose: it is what the scalar tracker filled its matrix with,
    and the Hungarian tie-breaks depend on it.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    denom = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((inter > 0.0) & (denom > 0.0), inter / denom, 0.0)
    return out.astype(np.float32)


def _greedy(score: NDArray, valid: NDArray, descending: bool) -> Tuple[List[Tuple[int, int]], set, set]:
    """Greedy one-to-one over ``valid`` cells, best score first, ties in (det, track) order."""
    di, ti = np.nonzero(valid)
    if len(di) == 0:
        return [], set(), set()
    s = score[di, ti]
    order = np.argsort(-s if descending else s, kind="stable")
    det_used, trk_used, matches = set(), set(), []
    for k in order:
        d, t = int(di[k]), int(ti[k])
        if d in det_used or t in trk_used:
            continue
        det_used.add(d)
        trk_used.add(t)
        matches.append((d, t))
    return matches, det_used, trk_used


def _hungarian(iou: NDArray[np.float32], iou_threshold: float) -> Tuple[List[Tuple[int, int]], set, set]:
    from scipy.optimize import linear_sum_assignment

    cost = 1.0 - iou
    cost[iou < iou_threshold] = _LARGE
    rows, cols = linear_sum_assignment(cost)
    matches, det_used, trk_used = [], set(), set()
    for d, t in zip(rows, cols):
        if iou[d, t] >= iou_threshold and cost[d, t] < _LARGE:
            matches.append((int(d), int(t)))
            det_used.add(int(d))
            trk_used.add(int(t))
    return matches, det_used, trk_used


def _center_dist_sq(a: NDArray, b: NDArray) -> NDArray:
    ca = 0.5 * (a[:, :2] + a[:, 2:4])
    cb = 0.5 * (b[:, :2] + b[:, 2:4])
    d = ca[:, None, :] - cb[None, :, :]
    return d[..., 0] ** 2 + d[..., 1] ** 2


def track_boxes(frames: ArrayLike, boxes: ArrayLike, classes: Optional[ArrayLike] = None,
                mode: str = "hungarian", iou_threshold: float = 0.3, class_aware: bool = True,
                max_age: int = -1, max_center_distance: float = 10.0) -> NDArray[np.int64]:
    """Assign a track id to every detection.

    :param frames: ``(N,)`` frame index of each detection
    :param boxes: ``(N, 4)`` xyxy or ``(N, 6)`` ``[x1 y1 z1 x2 y2 z2]``
    :param classes: ``(N,)`` class id per detection (``None`` = single class)
    :param mode: ``"greedy"`` / ``"hungarian"`` on IoU, ``"center"`` greedy on
        centre distance, ``"hungarian_center"`` IoU first then centre fallback
    :param iou_threshold: minimum IoU to accept an overlap match
    :param class_aware: never join detections of different classes
    :param max_age: frames a track may go unmatched before it is closed
        (``-1`` = never)
    :param max_center_distance: gate for the centre-distance modes, in box units
    :return: ``(N,)`` track ids starting at 1, in order of creation; aligned
        with the input
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    fr = np.asarray(frames).astype(np.int64).ravel()
    xyxy, _ = split_boxes(boxes)
    n = len(fr)
    if xyxy.shape[0] != n:
        raise ValueError(f"frames ({n}) and boxes ({xyxy.shape[0]}) differ in length")
    cls = np.zeros(n, dtype=np.int64) if classes is None else np.asarray(classes).astype(np.int64).ravel()
    if len(cls) != n:
        raise ValueError("classes must be (N,)")
    if classes is None:
        class_aware = False

    ids = np.zeros(n, dtype=np.int64)
    # live tracks: parallel python lists, in creation/survival order like the original
    trk_box: List[NDArray] = []
    trk_cls: List[int] = []
    trk_age: List[int] = []
    trk_id: List[int] = []
    next_id = 1
    max_d2 = float(max_center_distance) ** 2

    order = np.argsort(fr, kind="stable")          # frame ascending, input order within a frame
    fr_sorted = fr[order]
    bounds = np.flatnonzero(np.diff(fr_sorted)) + 1
    for det_idx in (np.split(order, bounds) if n else []):
        d_box = xyxy[det_idx]
        d_cls = cls[det_idx]
        nd, nt = len(det_idx), len(trk_box)
        matches: List[Tuple[int, int]] = []
        det_used: set = set()
        trk_used: set = set()
        if nd and nt:
            t_box = np.vstack(trk_box)
            t_cls = np.asarray(trk_cls)
            allowed = np.ones((nd, nt), dtype=bool)
            if class_aware:
                allowed = d_cls[:, None] == t_cls[None, :]
            if mode in ("greedy", "hungarian", "hungarian_center"):
                iou = iou_matrix(d_box, t_box)
                iou[~allowed] = 0.0
                if mode == "greedy":
                    matches, det_used, trk_used = _greedy(iou, allowed & (iou >= iou_threshold), descending=True)
                else:
                    matches, det_used, trk_used = _hungarian(iou, iou_threshold)
                if mode == "hungarian_center":
                    rem_d = [i for i in range(nd) if i not in det_used]
                    rem_t = [i for i in range(nt) if i not in trk_used]
                    if rem_d and rem_t:
                        d2 = _center_dist_sq(d_box[rem_d], t_box[rem_t])
                        ok = allowed[np.ix_(rem_d, rem_t)] & (d2 <= max_d2)
                        m2, _, _ = _greedy(d2, ok, descending=False)
                        for a, b in m2:
                            matches.append((rem_d[a], rem_t[b]))
                            det_used.add(rem_d[a])
                            trk_used.add(rem_t[b])
            else:  # center
                d2 = _center_dist_sq(d_box, t_box)
                matches, det_used, trk_used = _greedy(d2, allowed & (d2 <= max_d2), descending=False)

        for d, t in matches:
            trk_box[t] = d_box[d]
            trk_age[t] = 0
            ids[det_idx[d]] = trk_id[t]
        for d in range(nd):
            if d in det_used:
                continue
            trk_box.append(d_box[d])
            trk_cls.append(int(d_cls[d]))
            trk_age.append(0)
            trk_id.append(next_id)
            ids[det_idx[d]] = next_id
            next_id += 1
        keep = []
        for t in range(nt):
            if t not in trk_used:
                trk_age[t] += 1
            if max_age < 0 or trk_age[t] <= max_age:
                keep.append(t)
        keep += list(range(nt, len(trk_box)))          # tracks created this frame
        trk_box = [trk_box[t] for t in keep]
        trk_cls = [trk_cls[t] for t in keep]
        trk_age = [trk_age[t] for t in keep]
        trk_id = [trk_id[t] for t in keep]
    return ids


@dataclass(frozen=True)
class Tracks:
    """Track rows as parallel arrays, sorted by ``(frame, track_id)``.

    ``source`` indexes the detection a row came from, ``-1`` for
    interpolated rows.
    """
    frames: NDArray[np.int64]
    track_ids: NDArray[np.int64]
    boxes: NDArray[np.float64]           # (N, 4) or (N, 6), same width as the input
    confidences: NDArray[np.float64]
    classes: NDArray[np.int64]
    interpolated: NDArray[np.bool_]
    source: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def n_tracks(self) -> int:
        return len(np.unique(self.track_ids)) if len(self.track_ids) else 0


def interpolate_tracks(frames: ArrayLike, track_ids: ArrayLike, boxes: ArrayLike,
                       confidences: Optional[ArrayLike] = None, classes: Optional[ArrayLike] = None,
                       fill: bool = True, confidence: str = "mean") -> Tracks:
    """Sort rows by ``(frame, track_id)`` and, when ``fill``, close every gap
    inside a track with linearly interpolated boxes.

    :param confidence: how an interpolated row's confidence is set -
        ``"mean"`` of the two neighbours (the QGIS plugin) or ``"linear"``
        between them (``bambi.georeferenced_tracking``)
    :return: :class:`Tracks`
    """
    if confidence not in ("mean", "linear"):
        raise ValueError("confidence must be 'mean' or 'linear'")
    fr = np.asarray(frames).astype(np.int64).ravel()
    tid = np.asarray(track_ids).astype(np.int64).ravel()
    b = np.asarray(boxes, dtype=np.float64)
    n = len(fr)
    if b.ndim != 2 or b.shape[0] != n or len(tid) != n:
        raise ValueError("frames, track_ids and boxes must have the same length")
    conf = np.ones(n) if confidences is None else np.asarray(confidences, dtype=np.float64).ravel()
    cls = np.zeros(n, dtype=np.int64) if classes is None else np.asarray(classes).astype(np.int64).ravel()

    order = np.lexsort((tid, fr))
    out_f = [fr[order]]
    out_t = [tid[order]]
    out_b = [b[order]]
    out_c = [conf[order]]
    out_k = [cls[order]]
    out_i = [np.zeros(n, dtype=bool)]
    out_s = [order.astype(np.int64)]
    if fill and n:
        for t in np.unique(tid):
            rows = np.flatnonzero(tid == t)
            rows = rows[np.argsort(fr[rows], kind="stable")]
            f = fr[rows]
            gaps = np.diff(f)
            for k in np.flatnonzero(gaps > 1):
                g = int(gaps[k])
                a, z = rows[k], rows[k + 1]
                w = np.arange(1, g, dtype=np.float64) / g                       # (g-1,)
                out_f.append(f[k] + np.arange(1, g, dtype=np.int64))
                out_t.append(np.full(g - 1, t, dtype=np.int64))
                out_b.append(b[a][None, :] + w[:, None] * (b[z] - b[a])[None, :])
                if confidence == "mean":
                    out_c.append(np.full(g - 1, 0.5 * (conf[a] + conf[z])))
                    out_k.append(np.full(g - 1, cls[z], dtype=np.int64))    # plugin: class of the later row
                else:
                    out_c.append(conf[a] + w * (conf[z] - conf[a]))
                    out_k.append(np.full(g - 1, cls[a], dtype=np.int64))    # engine: class of the earlier row
                out_i.append(np.ones(g - 1, dtype=bool))
                out_s.append(np.full(g - 1, -1, dtype=np.int64))
    F = np.concatenate(out_f)
    T = np.concatenate(out_t)
    B = np.vstack(out_b) if n else b.reshape(0, b.shape[1])
    C = np.concatenate(out_c)
    K = np.concatenate(out_k)
    I = np.concatenate(out_i)
    S = np.concatenate(out_s)
    o = np.lexsort((T, F))
    return Tracks(F[o], T[o], B[o], C[o], K[o], I[o], S[o])
