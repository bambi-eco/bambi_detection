# -*- coding: utf-8 -*-
"""The valid area of a frame, in pixels and on the ground.

Undistorted frames carry an invalid border; the extractors write a mask
image (``mask_T.png`` / ``mask_W.png``) that is white where the frame is
usable. Its outline - :func:`mask_polygon` - is what the plugin's FoV and
GeoTIFF steps project onto the DEM: the footprint of the *content*, not of
the frame rectangle, so a GeoTIFF's bounds and the FoV polygon agree with
the rendered picture.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["mask_polygon", "default_mask_polygon", "polygon_to_world", "polygon_bounds"]


def mask_polygon(mask: ArrayLike, simplify_epsilon: float = 2.0) -> Optional[NDArray[np.float64]]:
    """Outline of the largest white region of a mask image -> ``(K, 2)`` pixels.

    :param mask: ``(H, W)`` or ``(H, W, C)`` image; > 127 counts as valid
    :param simplify_epsilon: Douglas-Peucker tolerance in pixels (0 = keep every vertex)
    :return: the polygon, or ``None`` when the mask has no white region
    """
    import cv2

    m = np.asarray(mask)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if m.shape[2] == 3 else m[:, :, 0]
    _, binary = cv2.threshold(m.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if simplify_epsilon > 0:
        largest = cv2.approxPolyDP(largest, float(simplify_epsilon), True)
    return np.asarray(largest, dtype=np.float64).reshape(-1, 2)


def default_mask_polygon(width: int, height: int) -> NDArray[np.float64]:
    """The whole frame as the plugin's 8-point polygon (corners + edge midpoints)."""
    w, h = int(width), int(height)
    return np.array([[0, 0], [w // 2, 0], [w, 0], [w, h // 2], [w, h], [w // 2, h], [0, h], [0, h // 2]], dtype=np.float64)


def polygon_to_world(polygon: ArrayLike, camera, width: int, height: int, mesh,
                     legacy: bool = True) -> NDArray[np.float64]:
    """Where a pixel polygon lands on the terrain -> ``(K, 3)`` DEM-local, NaN for misses.

    ``legacy=True`` reproduces the plugin's recipe exactly: each vertex is
    cast as the four corners of the one-pixel box ``[x, x+1] x [y, y+1]``
    (truncated to int) and the hits are averaged - so bounds derived here
    equal the plugin's GeoTIFF/FoV bounds. ``legacy=False`` casts the vertex
    itself, sub-pixel.
    """
    from bambi.geo.georef import pixels_to_world, pixels_to_world_legacy

    poly = np.atleast_2d(np.asarray(polygon, dtype=np.float64))[:, :2]
    out = np.full((len(poly), 3), np.nan)
    if not legacy:
        return pixels_to_world(poly, camera, width, height, mesh)
    for i, (px, py) in enumerate(poly):
        pts = pixels_to_world_legacy([px, px + 1, px + 1, px], [py, py, py + 1, py + 1], width, height, mesh, camera)
        if len(pts):
            out[i] = pts.mean(axis=0)
    return out


def polygon_bounds(points: ArrayLike) -> Optional[tuple]:
    """``(min_x, min_y, max_x, max_y)`` of the finite rows of ``(K, 2|3)``; ``None`` below 3 hits."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    p = p[np.isfinite(p[:, :2]).all(axis=1)]
    if len(p) < 3:
        return None
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))
