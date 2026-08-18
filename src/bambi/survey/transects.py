# -*- coding: utf-8 -*-
"""The flight path as geometry, and how it is cut into transects.

A *transect* is a contiguous frame range ``[start, end]`` of one modality's
frames. Its length is measured along the flight path - the polyline through
the per-frame camera ground positions (``Poses.positions[:, :2]``, DEM-local
metres, so distances are metres without any CRS handling). Everything here is
``(N,)`` per frame or ``(K, 2)`` per transect; the transect *store* (json /
csv files, names, ids) stays with the tool that edits them, in the plugin.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["cumulative_distances", "path_length", "frame_after_distance", "centerline",
           "split_by_distance", "split_into", "transect_lengths"]


def cumulative_distances(positions: ArrayLike) -> NDArray[np.float64]:
    """Distance flown up to every frame, metres. ``(N, 2|3)`` positions -> ``(N,)``.

    Only x/y count (transect lengths are horizontal ground distances). Frames
    without a position (NaN) contribute no distance and inherit the previous
    value, so the result always has one entry per frame and never decreases.
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("positions must be (N, 2) or (N, 3)")
    xy = p[:, :2]
    ok = np.isfinite(xy).all(axis=1)
    cum = np.zeros(len(xy))
    if ok.sum() >= 2:
        idx = np.flatnonzero(ok)
        steps = np.hypot(*np.diff(xy[idx], axis=0).T)
        cum[idx[1:]] = np.cumsum(steps)
    # frames without a position inherit the last known value
    last = np.maximum.accumulate(np.where(ok, np.arange(len(xy)), -1))
    return np.where(last >= 0, cum[np.maximum(last, 0)], 0.0)


def path_length(cum: ArrayLike, frame_a: int, frame_b: int) -> float:
    """Metres flown between two frames (either order), from :func:`cumulative_distances`."""
    c = np.asarray(cum, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    a = int(np.clip(frame_a, 0, len(c) - 1))
    b = int(np.clip(frame_b, 0, len(c) - 1))
    return float(abs(c[b] - c[a]))


def frame_after_distance(cum: ArrayLike, start_frame: int, meters: float) -> int:
    """First frame at or beyond ``meters`` of flight after ``start_frame``, or ``-1``
    when the remaining flight path is shorter than that."""
    c = np.asarray(cum, dtype=np.float64)
    if len(c) == 0 or meters < 0:
        return -1
    start = int(np.clip(start_frame, 0, len(c) - 1))
    idx = int(np.searchsorted(c, c[start] + meters, side="left"))
    idx = max(idx, start)
    return idx if idx < len(c) else -1


def centerline(positions: ArrayLike, first_frame: int, last_frame: int) -> NDArray[np.float64]:
    """The flight path of one transect as an ``(M, 2)`` polyline (frames
    without a position skipped; either frame order)."""
    p = np.asarray(positions, dtype=np.float64)[:, :2]
    lo = max(0, min(int(first_frame), int(last_frame)))
    hi = min(len(p) - 1, max(int(first_frame), int(last_frame)))
    seg = p[lo:hi + 1]
    return seg[np.isfinite(seg).all(axis=1)]


def split_by_distance(cum: ArrayLike, meters: float, start_frame: int = 0, end_frame: int = -1
                      ) -> NDArray[np.int64]:
    """Cut the flight into consecutive transects of ``meters`` each.

    :return: ``(K, 2)`` inclusive ``[start, end]`` frame ranges; the last one
        takes whatever is left (it may be shorter than ``meters``)
    """
    c = np.asarray(cum, dtype=np.float64)
    if len(c) == 0 or meters <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    end = len(c) - 1 if end_frame < 0 else min(int(end_frame), len(c) - 1)
    ranges = []
    s = int(np.clip(start_frame, 0, end))
    while s <= end:
        e = frame_after_distance(c[:end + 1], s, meters)
        if e < 0 or e >= end:
            ranges.append((s, end))
            break
        e = max(e, s + 1) if e == s else e
        ranges.append((s, e))
        s = e + 1
    return np.asarray(ranges, dtype=np.int64).reshape(-1, 2)


def split_into(cum: ArrayLike, n_transects: int) -> NDArray[np.int64]:
    """Cut the flight into ``n`` transects of (nearly) equal length -> ``(n, 2)`` frame ranges."""
    c = np.asarray(cum, dtype=np.float64)
    n = int(n_transects)
    if len(c) == 0 or n <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    total = float(c[-1])
    if total <= 0:
        return np.array([[0, len(c) - 1]] + [[len(c) - 1, len(c) - 1]] * (n - 1), dtype=np.int64)
    edges = np.searchsorted(c, np.linspace(0, total, n + 1)[1:-1], side="left")
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges - 1, [len(c) - 1]])
    ends = np.maximum(ends, starts)
    return np.column_stack([starts, ends]).astype(np.int64)


def transect_lengths(cum: ArrayLike, ranges: ArrayLike) -> NDArray[np.float64]:
    """Length in metres of every ``(K, 2)`` frame range."""
    r = np.asarray(ranges).reshape(-1, 2)
    return np.array([path_length(cum, a, b) for a, b in r], dtype=np.float64)
