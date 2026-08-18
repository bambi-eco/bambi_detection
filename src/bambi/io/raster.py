# -*- coding: utf-8 -*-
"""The file edge for single-band rasters (density / coverage GeoTIFFs)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

PathLike = Union[str, Path]

__all__ = ["write_single_band", "read_single_band"]


def write_single_band(path: PathLike, values: ArrayLike, bounds: Tuple[float, float, float, float],
                      epsg: int, nodata: Optional[float] = None, description: str = "",
                      dtype: Optional[str] = None) -> Path:
    """Write ``(H, W)`` values as a GeoTIFF over world ``bounds`` (row 0 = north).

    LZW-compressed, tiled when larger than 256 px; the CRS is written from
    the EPSG code. Overviews are left to the consumer.
    """
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.asarray(values)
    if dtype:
        arr = arr.astype(dtype)
    height, width = arr.shape
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "dtype": arr.dtype.name, "width": width, "height": height, "count": 1,
               "transform": from_bounds(*bounds, width, height), "compress": "lzw",
               "crs": rasterio.crs.CRS.from_epsg(int(epsg))}
    if nodata is not None:
        profile["nodata"] = nodata
    if width > 256 and height > 256:
        profile.update({"tiled": True, "blockxsize": 256, "blockysize": 256})
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(arr, 1)
        if description:
            dst.set_band_description(1, description)
    return p


def read_single_band(path: PathLike):
    """``(values (H, W), bounds, cell_size, epsg or None, nodata)`` of band 1."""
    import rasterio

    with rasterio.open(str(path)) as src:
        arr = src.read(1)
        b = src.bounds
        epsg = src.crs.to_epsg() if src.crs else None
        return arr, (b.left, b.bottom, b.right, b.top), abs(src.transform.a), epsg, src.nodata
