# -*- coding: utf-8 -*-
"""Density surfaces and coverage grids, as rasters.

* :func:`kde_grid` - a kernel-density estimate of animal locations: points
  are binned on a grid, smoothed with a Gaussian of ``bandwidth`` metres and
  scaled to *animals per hectare*. Same recipe as the QGIS plugin's density
  heatmap step (padding of three bandwidths, size cap, no-signal threshold),
  so a raster written from this grid equals the plugin's.
* :func:`coverage_grid` - how many frames saw each cell, from the frames'
  ground footprints. (The plugin counts overlaps of the exported GeoTIFFs
  instead; that needs the renders. Footprints give the same map without
  them, and it is what the analytics really ask - "how often was this
  ground looked at".)

Grids follow raster convention: row 0 is the northern edge, ``bounds`` is
``(min_x, min_y, max_x, max_y)`` in the points' CRS.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Grid", "kde_grid", "coverage_grid", "grid_extent", "NODATA"]

#: The plugin's no-signal value in density rasters.
NODATA = -9999.0
#: Largest raster side the plugin will produce; the cell size grows to fit.
MAX_DIM = 8192


class Grid(NamedTuple):
    """A raster: ``values (H, W)`` + world ``bounds`` + ``cell_size`` (m)."""
    values: NDArray
    bounds: Tuple[float, float, float, float]
    cell_size: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    def cell_of(self, xy: ArrayLike) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(row, col)`` of world points (may fall outside the grid)."""
        p = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        col = np.floor((p[:, 0] - self.bounds[0]) / self.cell_size).astype(np.int64)
        row = np.floor((self.bounds[3] - p[:, 1]) / self.cell_size).astype(np.int64)
        return row, col


def grid_extent(points: ArrayLike, cell_size: float, pad: float = 0.0,
                max_dim: int = MAX_DIM) -> Tuple[Tuple[float, float, float, float], float, int, int]:
    """Bounds, (possibly enlarged) cell size and ``(width, height)`` for a grid
    covering ``points`` padded by ``pad`` metres, capped at ``max_dim`` cells a side."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    if len(p) == 0:
        raise ValueError("no points")
    min_x, max_x = float(p[:, 0].min() - pad), float(p[:, 0].max() + pad)
    min_y, max_y = float(p[:, 1].min() - pad), float(p[:, 1].max() + pad)
    cs = float(cell_size)
    width = int(np.ceil((max_x - min_x) / cs))
    height = int(np.ceil((max_y - min_y) / cs))
    if width > max_dim or height > max_dim:
        cs = max((max_x - min_x) / max_dim, (max_y - min_y) / max_dim)
        width = int(np.ceil((max_x - min_x) / cs))
        height = int(np.ceil((max_y - min_y) / cs))
    return (min_x, min_y, max_x, max_y), cs, max(1, width), max(1, height)


def kde_grid(points: ArrayLike, cell_size: float = 5.0, bandwidth: float = 25.0,
             nodata: Optional[float] = NODATA) -> Grid:
    """Kernel-density estimate of point locations in **points per hectare**.

    :param points: ``(N, 2)`` ground positions (metric CRS)
    :param cell_size: raster cell, metres
    :param bandwidth: Gaussian sigma, metres
    :param nodata: value for cells with no signal (below ~1e-4 of one point's
        contribution to a cell); ``None`` keeps the raw density everywhere
    :return: :class:`Grid` of float32
    """
    from scipy.ndimage import gaussian_filter

    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    if len(p) < 1:
        raise ValueError("no points")
    bounds, cs, width, height = grid_extent(p, cell_size, pad=3.0 * bandwidth)
    min_x, min_y, max_x, max_y = bounds
    counts = np.zeros((height, width), dtype=np.float64)
    col = ((p[:, 0] - min_x) / cs).astype(np.int64)                # int() truncation like the plugin
    row = ((max_y - p[:, 1]) / cs).astype(np.int64)
    ok = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    np.add.at(counts, (row[ok], col[ok]), 1.0)
    smoothed = gaussian_filter(counts, sigma=bandwidth / cs, mode="constant", cval=0.0)
    cell_area = cs * cs
    density = smoothed / cell_area * 10000.0
    if nodata is None:
        return Grid(density.astype(np.float32), bounds, cs)
    eps = (1.0 / cell_area * 10000.0) * 1e-4
    return Grid(np.where(density > eps, density, nodata).astype(np.float32), bounds, cs)


def coverage_grid(footprints: Sequence[ArrayLike], cell_size: float = 1.0,
                  bounds: Optional[Tuple[float, float, float, float]] = None) -> Grid:
    """How many footprints cover each cell.

    :param footprints: sequence of ``(K, 2|3)`` ground polygons (one per frame);
        NaN rows are dropped, polygons with fewer than 3 finite corners skipped
    :param cell_size: raster cell, metres
    :param bounds: ``(min_x, min_y, max_x, max_y)``; default the polygons' extent
    :return: :class:`Grid` of uint16 counts (0 = never seen)
    """
    from matplotlib.path import Path as MplPath

    polys = []
    for fp in footprints:
        q = np.atleast_2d(np.asarray(fp, dtype=np.float64))[:, :2]
        q = q[np.isfinite(q).all(axis=1)]
        if len(q) >= 3:
            polys.append(q)
    if not polys:
        raise ValueError("no usable footprints")
    if bounds is None:
        allp = np.vstack(polys)
        bounds, cs, width, height = grid_extent(allp, cell_size)
    else:
        cs = float(cell_size)
        width = max(1, int(np.ceil((bounds[2] - bounds[0]) / cs)))
        height = max(1, int(np.ceil((bounds[3] - bounds[1]) / cs)))
    min_x, min_y, max_x, max_y = bounds
    count = np.zeros((height, width), dtype=np.uint16)
    xs = min_x + (np.arange(width) + 0.5) * cs
    ys = max_y - (np.arange(height) + 0.5) * cs
    for q in polys:
        c0 = max(0, int(np.floor((q[:, 0].min() - min_x) / cs)))
        c1 = min(width, int(np.ceil((q[:, 0].max() - min_x) / cs)) + 1)
        r0 = max(0, int(np.floor((max_y - q[:, 1].max()) / cs)))
        r1 = min(height, int(np.ceil((max_y - q[:, 1].min()) / cs)) + 1)
        if c1 <= c0 or r1 <= r0:
            continue
        gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
        inside = MplPath(q).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gy.shape)
        count[r0:r1, c0:c1] += inside.astype(np.uint16)
    return Grid(count, bounds, cs)
