# -*- coding: utf-8 -*-
"""Elevation grids -> triangle meshes, as arrays.

The pipeline's DEM meshes (``*_dem.glb`` written by the Upper-Austria
downloader and by the public Dataset's ``dem_from_poses.py``) all share one
layout: one vertex per raster cell, x = column * cell width, y counted
*upwards* from the bottom row, z = elevation minus the DEM's base elevation,
two triangles per cell. :func:`heightfield_mesh` is that layout, vectorised,
so a mesh built here is index-for-index the one those tools write.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["heightfield_mesh", "heightfield_grid_xy", "sample_heightfield"]


def heightfield_grid_xy(shape: Tuple[int, int], cell_size: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mesh-local x/y of every cell of an ``(H, W)`` grid.

    :param shape: ``(rows, cols)`` of the elevation grid
    :param cell_size: ``(cell_width, cell_height)`` metres; positive
    :return: ``xs (W,)``, ``ys (H,)`` - ``ys[0]`` is the *top* row, which sits at
        the largest y (raster row order runs north to south)
    """
    rows, cols = int(shape[0]), int(shape[1])
    cw, ch = (float(v) for v in np.abs(np.asarray(cell_size, dtype=np.float64)).ravel()[:2])
    xs = np.arange(cols, dtype=np.float64) * cw
    ys = (rows - 1 - np.arange(rows, dtype=np.float64)) * ch
    return xs, ys


def heightfield_mesh(elevation: ArrayLike, cell_size: ArrayLike,
                     base_elevation: float | None = None) -> Tuple[NDArray[np.float32], NDArray[np.uint32], float]:
    """Triangulate an elevation grid the way the pipeline's DEM tools do.

    :param elevation: ``(H, W)`` metres; NaN = no data (filled with the base)
    :param cell_size: ``(cell_width, cell_height)`` metres
    :param base_elevation: z-origin of the mesh; default the minimum valid
        elevation (what the DEM tools use)
    :return: ``vertices (H*W, 3) float32``, ``faces (2*(H-1)*(W-1), 3) uint32``,
        and the base elevation actually used
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
        raise ValueError(f"elevation must be a (H, W) grid with H, W >= 2, got {z.shape}")
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("elevation has no valid (finite) cells")
    base = float(np.nanmin(z)) if base_elevation is None else float(base_elevation)
    z = np.where(valid, z, base) - base

    rows, cols = z.shape
    xs, ys = heightfield_grid_xy(z.shape, cell_size)
    gx, gy = np.meshgrid(xs, ys)                                    # (H, W) each
    vertices = np.stack([gx, gy, z], axis=-1).reshape(-1, 3).astype(np.float32)

    r = np.arange(rows - 1)[:, None]
    c = np.arange(cols - 1)[None, :]
    i00 = (r * cols + c).ravel()
    i10 = i00 + 1
    i01 = i00 + cols
    i11 = i01 + 1
    faces = np.stack([np.stack([i00, i01, i10], axis=1),
                      np.stack([i10, i01, i11], axis=1)], axis=1).reshape(-1, 3).astype(np.uint32)
    return vertices, faces, base


def sample_heightfield(elevation: ArrayLike, cell_size: ArrayLike, xy: ArrayLike,
                       base_elevation: float | None = None) -> NDArray[np.float64]:
    """Mesh-local z at mesh-local ``xy`` points, exactly on the triangulated
    surface :func:`heightfield_mesh` builds (NaN outside the grid).

    Piecewise-linear on the same two triangles per cell, so a point sampled
    here lies *on* the mesh - which is what a synthetic marker needs. The
    ray-caster in :mod:`bambi.geo.georef` is the authority for real work.
    """
    z = np.asarray(elevation, dtype=np.float64)
    base = float(np.nanmin(z)) if base_elevation is None else float(base_elevation)
    z = np.where(np.isfinite(z), z, base) - base
    rows, cols = z.shape
    cw, ch = (float(v) for v in np.abs(np.asarray(cell_size, dtype=np.float64)).ravel()[:2])
    pts = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    fx = pts[:, 0] / cw
    fy = (rows - 1) - pts[:, 1] / ch                        # row index grows downwards
    out = np.full(len(pts), np.nan)
    ok = (fx >= 0) & (fx <= cols - 1) & (fy >= 0) & (fy <= rows - 1)
    x0 = np.clip(np.floor(fx[ok]).astype(int), 0, cols - 2)
    y0 = np.clip(np.floor(fy[ok]).astype(int), 0, rows - 2)
    tx = fx[ok] - x0
    ty = fy[ok] - y0
    z00, z10, z01, z11 = z[y0, x0], z[y0, x0 + 1], z[y0 + 1, x0], z[y0 + 1, x0 + 1]
    # cell diagonal runs from (row+1, col) to (row, col+1): faces (i00,i01,i10) and (i10,i01,i11)
    lower = tx + ty <= 1.0
    out[ok] = np.where(lower,
                       z00 + tx * (z10 - z00) + ty * (z01 - z00),
                       z11 + (1 - tx) * (z01 - z11) + (1 - ty) * (z10 - z11))
    return out
