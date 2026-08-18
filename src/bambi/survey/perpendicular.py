# -*- coding: utf-8 -*-
"""Perpendicular distances from ground positions to the flight route.

Line-transect distance sampling wants, for every animal, its perpendicular
distance to the transect line the observer flew. Two ways to find the foot
point are kept, both from the QGIS plugin:

* :func:`nearest_on_polyline` - the closest point on the whole route, with
  the projection *clamped* to each segment (a point beyond the end of the
  route measures to the end point). This is what transect assignment uses.
* :func:`perpendicular_to_route` - the plugin's "Calculate Perpendicular"
  rule: pick the route segment the *camera* was flying on (nearest to the
  drone position at that frame), then drop a true perpendicular onto that
  segment's infinite line. This never snaps a detection to a neighbouring,
  parallel transect - which the nearest-point rule happily does on a lawn-
  mower pattern - and always returns a line at 90 degrees to the route.

Plus :func:`flat_footprint`, the flat-earth frame footprint (a rotated
rectangle under the camera) the plugin uses where no DEM footprint exists.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["nearest_on_polyline", "nearest_segments", "perpendicular_to_route", "flat_footprint",
           "route_length"]


def _segments(line: ArrayLike) -> Tuple[NDArray, NDArray, NDArray]:
    r = np.asarray(line, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2 or len(r) < 1:
        raise ValueError("route must be (M, 2|3) with M >= 1")
    r = r[:, :2]
    if len(r) == 1:
        r = np.vstack([r, r])
    a = r[:-1]
    d = r[1:] - a
    return a, d, np.einsum("ij,ij->i", d, d)


def route_length(line: ArrayLike) -> float:
    """Horizontal length of a polyline in metres."""
    a, d, _ = _segments(line)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def nearest_on_polyline(points: ArrayLike, line: ArrayLike
                        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Closest point on the route to each query point (projection clamped to segments).

    :param points: ``(N, 2|3)`` query points (x, y used)
    :param line: ``(M, 2|3)`` route
    :return: ``foot (N, 2)``, ``distance (N,)``, ``segment (N,)`` index of the
        winning segment (first one on ties, like the scalar original)
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    a, d, len_sq = _segments(line)
    rel = p[:, None, :] - a[None, :, :]                                   # (N, M-1, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.einsum("nmk,mk->nm", rel, d) / len_sq[None, :]
    t = np.where(len_sq[None, :] < 1e-12, 0.0, np.clip(t, 0.0, 1.0))
    foot = a[None, :, :] + t[:, :, None] * d[None, :, :]
    dist = np.hypot(*(p[:, None, :] - foot).transpose(2, 0, 1))
    seg = np.argmin(dist, axis=1)
    idx = np.arange(len(p))
    return foot[idx, seg], dist[idx, seg], seg.astype(np.int64)


def nearest_segments(points: ArrayLike, line: ArrayLike) -> NDArray[np.int64]:
    """Index of the route segment nearest to each point (clamped distance)."""
    return nearest_on_polyline(points, line)[2]


def perpendicular_to_route(points: ArrayLike, line: ArrayLike, camera_positions: ArrayLike = None
                           ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perpendicular foot and distance of each point to the flight route.

    With ``camera_positions`` (``(N, 2|3)``, the drone position at the frame
    each point was seen in) the segment is the one nearest to the camera and
    the foot is the *unclamped* projection onto that segment's line - the
    plugin's rule. Without them, the nearest clamped point on the route.

    :return: ``foot (N, 2)``, ``distance (N,)``
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    if camera_positions is None:
        foot, dist, _ = nearest_on_polyline(p, line)
        return foot, dist
    cam = np.atleast_2d(np.asarray(camera_positions, dtype=np.float64))[:, :2]
    if cam.shape[0] != p.shape[0]:
        raise ValueError("camera_positions must have one row per point")
    a, d, len_sq = _segments(line)
    seg = nearest_on_polyline(cam, line)[2]
    sa, sd, sl = a[seg], d[seg], len_sq[seg]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.einsum("nk,nk->n", p - sa, sd) / sl
    degenerate = sl < 1e-12
    t = np.where(degenerate, 0.0, t)
    foot = sa + t[:, None] * sd
    dist = np.hypot(*(p - foot).T)
    return foot, dist


def flat_footprint(position: ArrayLike, rotation: ArrayLike, fovy: float,
                   aspect_ratio: float = 4.0 / 3.0) -> NDArray[np.float64]:
    """Flat-earth ground footprint of a frame as a ``(4, 2)`` quadrilateral.

    The camera altitude is the DEM-local ``z`` of the pose (at least 1 m) and
    the FOV rectangle is rotated by the heading (``rotation[2]``). A nadir
    approximation - what the plugin's perpendicular step used before it had
    the DEM footprints; kept because it needs no mesh.
    """
    pos = np.asarray(position, dtype=np.float64).ravel()
    rot = np.asarray(rotation, dtype=np.float64).ravel()
    alt = max(float(pos[2]), 1.0)
    half_h = alt * np.tan(np.deg2rad(float(fovy) / 2.0))
    fov_x = 2.0 * np.arctan(float(aspect_ratio) * np.tan(np.deg2rad(float(fovy) / 2.0)))
    half_w = alt * np.tan(fov_x / 2.0)
    local = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])
    yaw = np.deg2rad(float(rot[2]) % 360.0)
    c, s = np.cos(yaw), np.sin(yaw)
    rotm = np.array([[c, -s], [s, c]])
    return local @ rotm.T + pos[:2]
