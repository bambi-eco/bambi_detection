# -*- coding: utf-8 -*-
"""The file edge for rendered rasters: per-frame GeoTIFFs (with a world file
beside them), reading them back, and merging a folder of them into an
orthomosaic by averaging - the plugin's ``geotiffs_{m}/`` and
``orthomosaic_{m}/`` products.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

PathLike = Union[str, Path]
Bounds = Tuple[float, float, float, float]

__all__ = ["write_frame_geotiff", "write_world_file", "read_geotiff", "merge_average", "count_overlaps"]


def write_world_file(image_path: PathLike, bounds: Bounds, width: int, height: int) -> Path:
    """The ESRI world file (``.tfw`` / ``.pgw`` / ``.jgw`` / ``.wld``) for a raster over ``bounds``."""
    p = Path(image_path)
    ext = {".tif": ".tfw", ".tiff": ".tfw", ".png": ".pgw", ".jpg": ".jgw", ".jpeg": ".jgw"}.get(p.suffix.lower())
    wf = p.with_suffix(ext) if ext else Path(str(p) + ".wld")
    min_x, min_y, max_x, max_y = bounds
    sx = (max_x - min_x) / width
    sy = -(max_y - min_y) / height
    wf.write_text(f"{sx:.10f}\n0.0\n0.0\n{sy:.10f}\n{min_x + sx / 2:.10f}\n{max_y + sy / 2:.10f}\n", encoding="utf-8")
    return wf


def write_frame_geotiff(path: PathLike, image: ArrayLike, valid: Optional[ArrayLike], bounds: Bounds, epsg: int,
                        nodata: int = 0, world_file: bool = True) -> Path:
    """``(H, W[, C])`` uint8 image -> LZW GeoTIFF over ``bounds`` (world CRS); invalid pixels become ``nodata``."""
    import rasterio
    from rasterio.transform import from_bounds

    img = np.asarray(image)
    data = img[np.newaxis] if img.ndim == 2 else np.moveaxis(img, -1, 0)
    data = np.array(data)                                        # own copy: nodata is written into it
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        data[:, ~v] = nodata
    count, height, width = data.shape
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "dtype": data.dtype.name, "width": width, "height": height, "count": count,
               "transform": from_bounds(*bounds, width, height), "compress": "lzw", "nodata": nodata,
               "crs": rasterio.crs.CRS.from_epsg(int(epsg))}
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(data)
    if world_file:
        write_world_file(p, bounds, width, height)
    return p


def read_geotiff(path: PathLike) -> Tuple[NDArray, Bounds, Optional[int], Optional[float]]:
    """``(image (H, W[, C]), bounds, epsg, nodata)`` - bands moved last."""
    import rasterio

    with rasterio.open(str(path)) as src:
        data = src.read()
        b = src.bounds
        epsg = src.crs.to_epsg() if src.crs else None
        nodata = src.nodata
    img = data[0] if data.shape[0] == 1 else np.moveaxis(data, 0, -1)
    return img, (b.left, b.bottom, b.right, b.top), epsg, nodata


def merge_average(paths: Sequence[PathLike], out_path: PathLike, nodata: int = 0,
                  resolution: Optional[float] = None) -> Path:
    """Average-merge GeoTIFFs into one orthomosaic (the plugin's ``merge_orthomosaic_average``).

    ``rasterio.merge`` has no "average"; a first pass fixes the output grid,
    a second accumulates a float64 sum and a count of valid contributors per
    pixel, and the quotient is rounded back to the sources' dtype.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge

    res = (resolution, resolution) if resolution and resolution > 0 else None
    datasets = [rasterio.open(str(p)) for p in paths]
    try:
        src_dtype = datasets[0].dtypes[0]
        base, out_transform = rio_merge(datasets, method="first", nodata=nodata, res=res)
        out_h, out_w = base.shape[1], base.shape[2]
        del base
        count = np.zeros((out_h, out_w), dtype=np.float64)

        def _sum_valid(merged_data, new_data, merged_mask, new_mask, index=None, roff=0, coff=0, **kwargs):
            valid = ~new_mask
            np.add(merged_data, new_data, out=merged_data, where=valid, casting="unsafe")
            band0 = valid[0] if valid.ndim == 3 else valid
            h, w = band0.shape
            count[roff:roff + h, coff:coff + w] += band0
            if merged_mask.shape == valid.shape:
                merged_mask[valid] = False

        summed, _ = rio_merge(datasets, method=_sum_valid, nodata=nodata, dtype="float64", res=res)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(count[None] > 0, summed / count[None], nodata)
        avg = np.rint(avg).astype(src_dtype)
        meta = datasets[0].meta.copy()
        meta.update({"driver": "GTiff", "height": out_h, "width": out_w, "count": avg.shape[0],
                     "transform": out_transform, "compress": "lzw", "nodata": nodata, "tiled": True,
                     "BIGTIFF": "IF_SAFER"})
    finally:
        for d in datasets:
            d.close()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", **meta) as dst:
        dst.write(avg)
    return p


def count_overlaps(paths: Sequence[PathLike], nodata: int = 0, resolution: Optional[float] = None
                   ) -> Tuple[NDArray[np.uint16], Bounds]:
    """How many of the GeoTIFFs cover each pixel of their common grid (the plugin's coverage map)."""
    import rasterio
    from rasterio.merge import merge as rio_merge

    res = (resolution, resolution) if resolution and resolution > 0 else None
    datasets = [rasterio.open(str(p)) for p in paths]
    try:
        base, out_transform = rio_merge(datasets, method="first", nodata=nodata, res=res)
        out_h, out_w = base.shape[1], base.shape[2]
        del base
        count = np.zeros((out_h, out_w), dtype=np.float64)

        def _count(merged_data, new_data, merged_mask, new_mask, index=None, roff=0, coff=0, **kwargs):
            valid = ~new_mask
            band0 = valid[0] if valid.ndim == 3 else valid
            h, w = band0.shape
            count[roff:roff + h, coff:coff + w] += band0
            if merged_mask.shape == valid.shape:
                merged_mask[valid] = False

        rio_merge(datasets, method=_count, nodata=nodata, res=res)
    finally:
        for d in datasets:
            d.close()
    from rasterio.transform import array_bounds
    b = array_bounds(out_h, out_w, out_transform)
    return count.astype(np.uint16), (b[0], b[1], b[2], b[3])
