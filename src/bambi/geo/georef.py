# -*- coding: utf-8 -*-
"""Pixels -> world coordinates on a DEM, as arrays.

Given a camera (from :mod:`bambi.geo.camera`), a triangle mesh of the terrain
and pixel coordinates in that camera's frame, cast rays through the pixels
and return where they hit the ground - DEM-local ``[east, north, up]``.
Everything here is ``(N, ...)`` in and ``(N, ...)`` out; a ray that misses the
mesh yields **NaN in its row**, so results stay aligned with their inputs. The
pipeline's older ``label_to_world_coordinates`` silently *dropped* misses and
truncated pixels to ``int``; :func:`pixels_to_world_legacy` reproduces that
byte-for-byte for parity, and is not what new code should call.

Coordinates: pixels are top-left-origin ``[x, y]`` in the frame's own
resolution; the mesh must be in the same DEM-local metric frame as the
poses (that is what :mod:`bambi.io.dem` returns).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "pixels_to_world",
    "boxes_to_world",
    "footprint",
    "pixels_to_world_legacy",
    "MISS",
]

#: What a missed ray looks like in every result of this module.
MISS = np.nan


def pixels_to_world(pixels: ArrayLike, camera, width: int, height: int, mesh) -> NDArray[np.float64]:
    """Cast a ray through each pixel and return where it hits the mesh.

    :param pixels: ``(N, 2)`` or ``(2,)`` ``[x, y]`` in the camera's frame
    :param camera: alfspy ``Camera`` for that frame (:func:`bambi.geo.camera.camera_from_pose`)
    :param width: frame width in pixels
    :param height: frame height in pixels
    :param mesh: ``trimesh.Trimesh`` of the DEM in DEM-local metres
    :return: ``(N, 3)`` DEM-local ``[east, north, up]``; NaN rows for misses
    """
    from alfspy.core.convert.convert import pixel_to_world_coord

    px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    if px.shape[1] != 2:
        raise ValueError(f"pixels must be (N, 2) [x, y], got {px.shape}")
    out = np.full((len(px), 3), MISS, dtype=np.float64)
    if len(px) == 0:
        return out
    hits = pixel_to_world_coord(px[:, 0].tolist(), px[:, 1].tolist(), int(width), int(height),
                                mesh, camera, include_misses=True)
    for i, h in enumerate(hits):
        if h is not None:
            out[i] = np.asarray(h, dtype=np.float64)[:3]
    return out


def boxes_to_world(boxes: ArrayLike, camera, width: int, height: int, mesh) -> NDArray[np.float64]:
    """Ground footprint of pixel boxes: all four corners of each box.

    :param boxes: ``(N, 4)`` ``[x1, y1, x2, y2]``
    :return: ``(N, 4, 3)`` corners in the order TL, TR, BR, BL; NaN where a
        corner's ray missed
    """
    b = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    if b.shape[1] != 4:
        raise ValueError(f"boxes must be (N, 4) [x1, y1, x2, y2], got {b.shape}")
    corners = np.stack([
        b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]],
    ], axis=1)                                            # (N, 4, 2)
    flat = pixels_to_world(corners.reshape(-1, 2), camera, width, height, mesh)
    return flat.reshape(len(b), 4, 3)


def footprint(camera, width: int, height: int, mesh,
              samples_per_edge: int = 1) -> NDArray[np.float64]:
    """Ground polygon of a frame's field of view.

    Walks the frame border clockwise from the top-left corner and casts each
    border pixel to the ground. With ``samples_per_edge=1`` that is the four
    corners; higher values trace lens/terrain curvature.

    :return: ``(4 * samples_per_edge, 3)`` DEM-local points, NaN for misses
    """
    s = max(1, int(samples_per_edge))
    w, h = width - 1, height - 1
    # Parametrise each edge from 0 to 1 (excluding the end, which the next edge starts).
    u = np.linspace(0.0, 1.0, s, endpoint=False)
    top = np.column_stack([u * w, np.zeros(s)])                    # (0,0) -> (w,0)
    right = np.column_stack([np.full(s, w), u * h])                # (w,0) -> (w,h)
    bottom = np.column_stack([w - u * w, np.full(s, h)])           # (w,h) -> (0,h)
    left = np.column_stack([np.zeros(s), h - u * h])               # (0,h) -> (0,0)
    border = np.vstack([top, right, bottom, left])
    return pixels_to_world(border, camera, width, height, mesh)


def pixels_to_world_legacy(pixel_xs: Sequence[float], pixel_ys: Sequence[float],
                           width: int, height: int, mesh, camera) -> NDArray[np.float64]:
    """Byte-for-byte reproduction of the pipeline's ``label_to_world_coordinates``.

    Truncates every pixel to ``int`` and DROPS misses, so the result may be
    shorter than the input - and, because alfspy's ``include_misses=False``
    hands back trimesh's hit order, the rows are **not in input order** either
    (a min/max box survives that; a polygon does not). Exists only so parity
    with the QGIS plugin's ``georeferenced.txt`` can be asserted; do not build
    on it.
    """
    from alfspy.core.convert.convert import pixel_to_world_coord

    xs = [int(float(x)) for x in pixel_xs]
    ys = [int(float(y)) for y in pixel_ys]
    hits = pixel_to_world_coord(xs, ys, int(width), int(height), mesh, camera, include_misses=False)
    return np.reshape(np.asarray(hits, dtype=np.float64), (-1, 3)) if len(hits) else np.zeros((0, 3))
