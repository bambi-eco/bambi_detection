# -*- coding: utf-8 -*-
"""Orthophotos and light-field integrals through alfspy, on arrays.

A *shot* is one frame with its camera; an *orthographic camera* looks
straight down on a DEM-local extent; the renderer projects shots onto the
terrain mesh as seen by that camera. Two products:

* one shot -> an orthophoto of that frame (``integral=False``): the plugin's
  per-frame GeoTIFF export;
* many shots -> their additive integral, normalised by how many shots saw
  each pixel (``integral=True``): the airborne light-field (ALFS) image
  that lets the ground be seen through the canopy.

Poses go in as arrays (DEM-local position, ``[tilt, roll, heading]``,
fovy) and out come ``(H, W, 4)`` uint8 RGBA rasters plus their DEM-local
bounds; files are :mod:`bambi.io.geotiff`'s business. The shot rotation is
``quaternion_from_drone_pose`` - the same convention :mod:`bambi.geo.camera`
uses for ray casting, so a render and its footprint agree at any tilt (the
QGIS plugin's rendering steps still spell shots with the older ``'zyx'``
composition, which coincides with this at nadir).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["render_size", "make_shot", "make_shots", "ortho_camera", "render_orthographic", "tiles", "tile_bounds",
           "tile_camera", "erode_valid_mask", "mask_texture", "crop_to_content", "frame_orthophoto", "shot_rotation",
           "render_integral_tiled",
           "ROTATION_CONVENTIONS", "Bounds"]

Bounds = Tuple[float, float, float, float]      # (min_x, min_y, max_x, max_y) DEM-local


# ---------------------------------------------------------------- sizing / tiling

def render_size(width_m: float, height_m: float, ground_resolution: Optional[float] = None,
                fixed: Optional[Tuple[int, int]] = None, max_dim: Optional[int] = None) -> Tuple[int, int]:
    """``(width_px, height_px)`` for an extent: from metres per pixel (rounded
    up) or a fixed pixel size; ``max_dim`` caps the longer side keeping the aspect."""
    if fixed is not None:
        w, h = int(fixed[0]), int(fixed[1])
    else:
        if not ground_resolution or ground_resolution <= 0:
            raise ValueError(f"ground_resolution must be positive, got {ground_resolution!r}")
        w = int(math.ceil(width_m / ground_resolution))
        h = int(math.ceil(height_m / ground_resolution))
    w, h = max(1, w), max(1, h)
    if max_dim and (w > max_dim or h > max_dim):
        s = max_dim / float(max(w, h))
        w, h = max(1, int(w * s)), max(1, int(h * s))
    return w, h


def tiles(size: Tuple[int, int], max_tile: int) -> NDArray[np.int64]:
    """``(T, 4)`` ``[tx, ty, tw, th]`` pixel tiles (row-major) covering ``(width, height)``."""
    w, h = int(size[0]), int(size[1])
    m = max(1, int(max_tile))
    out = []
    for ty in range(0, h, m):
        for tx in range(0, w, m):
            out.append((tx, ty, min(m, w - tx), min(m, h - ty)))
    return np.asarray(out, dtype=np.int64).reshape(-1, 4)


def tile_bounds(bounds: Bounds, size: Tuple[int, int], tile: ArrayLike) -> Bounds:
    """World extent of one ``[tx, ty, tw, th]`` tile of a ``size`` raster over ``bounds`` (row 0 = north)."""
    min_x, min_y, max_x, max_y = bounds
    tx, ty, tw, th = (int(v) for v in tile)
    sx = (max_x - min_x) / size[0]
    sy = (max_y - min_y) / size[1]
    return (min_x + tx * sx, max_y - (ty + th) * sy, min_x + (tx + tw) * sx, max_y - ty * sy)


# ---------------------------------------------------------------- shots and cameras

def mask_texture(mask: ArrayLike):
    """A mask image (``(H, W)`` or ``(H, W, C)`` uint8) -> alfspy ``TextureData`` for the renderer."""
    from alfspy.core.rendering import CtxShot, TextureData

    return TextureData(CtxShot._cvt_img(np.asarray(mask)))


def _correction_transform(translation, rotation):
    from pyrr import Quaternion, Vector3
    from alfspy.core.geo.transform import Transform

    t = Vector3(np.asarray(translation, dtype=np.float32))
    r = Vector3(np.asarray(rotation, dtype=np.float32))
    return Transform(t, Quaternion.from_eulers(r))


ROTATION_CONVENTIONS = ("drone_pose", "zyx")


def shot_rotation(rotation: ArrayLike, convention: str = "drone_pose"):
    """The pose ``[tilt, roll, heading]`` as the shot's quaternion.

    ``"drone_pose"`` (default) is ``quaternion_from_drone_pose`` - heading about
    world up after the tilt, the convention the ray caster uses too.
    ``"zyx"`` is ``quaternion_from_eulers(radians, 'zyx')``: what the QGIS
    plugin's ALFS / GeoTIFF steps still build their shots with. The two agree
    at nadir; away from it the ``'zyx'`` shot leans the wrong way (its tilt
    enters with the opposite sign) - kept only to reproduce those outputs.
    """
    from alfspy.core.util.pyrrs import quaternion_from_drone_pose, quaternion_from_eulers

    if convention not in ROTATION_CONVENTIONS:
        raise ValueError(f"convention must be one of {ROTATION_CONVENTIONS}")
    rot = [float(v) % 360.0 for v in np.asarray(rotation, dtype=np.float64)[:3]]
    if convention == "zyx":
        return quaternion_from_eulers([np.deg2rad(v) for v in rot], "zyx")
    return quaternion_from_drone_pose(rot)


def make_shot(ctx, image: Union[str, ArrayLike], position: ArrayLike, rotation: ArrayLike, fovy: float,
              aspect_ratio: float = 1.0, translation_correction: ArrayLike = (0.0, 0.0, 0.0),
              rotation_correction: ArrayLike = (0.0, 0.0, 0.0), lazy: bool = True,
              convention: str = "drone_pose"):
    """One alfspy ``CtxShot``.

    :param image: file path or a BGR(A)/grey uint8 array (OpenCV layout)
    :param position: ``(3,)`` DEM-local metres
    :param rotation: ``(3,)`` ``[tilt, roll, heading]`` degrees
    :param translation_correction: ``(3,)`` metres, ``rotation_correction``
        ``(3,)`` radians - the per-flight correction, applied by the renderer
        itself (its ``CtxShot.get_correction``)
    :param lazy: load the texture on first use (keeps hundreds of shots cheap)
    :param convention: see :func:`shot_rotation`
    """
    from pyrr import Vector3
    from alfspy.core.rendering import CtxShot

    return CtxShot(ctx, image, Vector3(np.asarray(position, dtype=np.float32)), shot_rotation(rotation, convention),
                   float(fovy), float(aspect_ratio), _correction_transform(translation_correction, rotation_correction),
                   lazy=lazy)


def make_shots(ctx, images: Sequence[Union[str, ArrayLike]], poses, fovy: ArrayLike, aspect_ratio: float = 1.0,
               translation_corrections: Optional[ArrayLike] = None, rotation_corrections: Optional[ArrayLike] = None,
               indices: Optional[ArrayLike] = None, lazy: bool = True,
               convention: str = "drone_pose") -> List[object]:
    """Shots for (a subset of) poses; ``images[k]`` belongs to ``indices[k]``."""
    n = len(poses)
    idx = np.arange(n) if indices is None else np.asarray(indices, dtype=int)
    if len(images) != len(idx):
        raise ValueError("one image per selected pose")
    fov = np.broadcast_to(np.asarray(fovy, dtype=np.float64), (n,))
    tc = np.zeros((n, 3)) if translation_corrections is None else np.asarray(translation_corrections, float)
    rc = np.zeros((n, 3)) if rotation_corrections is None else np.asarray(rotation_corrections, float)
    return [make_shot(ctx, img, poses.positions[i], poses.rotations[i], fov[i], aspect_ratio, tc[i], rc[i], lazy,
                      convention) for img, i in zip(images, idx)]


def ortho_camera(bounds: Bounds, height: float):
    """A nadir orthographic alfspy ``Camera`` over a DEM-local extent at ``height`` metres."""
    from pyrr import Quaternion, Vector3
    from alfspy.core.rendering import Camera

    min_x, min_y, max_x, max_y = bounds
    return Camera(orthogonal=True, orthogonal_size=(max_x - min_x, max_y - min_y),
                  position=Vector3([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, float(height)], dtype="f4"),
                  rotation=Quaternion(), near=0.1, far=10000.0)


def tile_camera(global_camera, bounds: Bounds, size: Tuple[int, int], tile: ArrayLike):
    """The sub-camera for one tile of a large orthographic render (plugin ``create_tile_camera``)."""
    from pyrr import Vector3
    from alfspy.core.rendering import Camera

    tb = tile_bounds(bounds, size, tile)
    return Camera(orthogonal=True, orthogonal_size=(tb[2] - tb[0], tb[3] - tb[1]),
                  position=Vector3([(tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0,
                                    global_camera.transform.position.z], dtype="f4"),
                  rotation=global_camera.transform.rotation, near=global_camera.near, far=global_camera.far)


# ---------------------------------------------------------------- rendering

def render_orthographic(ctx, mesh_data, texture_data, shots: Union[object, Iterable[object]], bounds: Bounds,
                        size: Tuple[int, int], camera_height: Optional[float] = None, mask=None,
                        integral: bool = False, release_shots: bool = False,
                        auto_contrast: bool = True) -> NDArray[np.uint8]:
    """Project shots onto the terrain as seen from straight above.

    :param ctx: render context (:func:`bambi.util.render_context.make_render_context`)
    :param mesh_data, texture_data: the DEM (:func:`bambi.io.dem.read_render_data`)
    :param shots: one shot (orthophoto) or many (integral)
    :param bounds: DEM-local extent to render
    :param size: ``(width, height)`` pixels
    :param camera_height: DEM-local z of the orthographic camera; default 100 m above the mesh
    :param mask: :func:`mask_texture` of the frames' valid area, or ``None``
    :param integral: additive integral normalised by overlap (ALFS) instead of a single projection
    :return: ``(H, W, 4)`` uint8 RGBA; alpha 0 where nothing was projected
    """
    from alfspy.core.rendering import RenderResultMode, Resolution
    from alfspy.core.rendering.renderer import Renderer

    if camera_height is None:
        camera_height = float(np.asarray(mesh_data.vertices)[:, 2].max()) + 100.0
    cam = ortho_camera(bounds, camera_height)
    renderer = Renderer(Resolution(int(size[0]), int(size[1])), ctx, cam, mesh_data, texture_data)
    try:
        if integral:
            shot_list = list(shots) if isinstance(shots, Iterable) else [shots]
            img = renderer.render_integral(shot_list, mask=mask, save=False, release_shots=release_shots,
                                           auto_contrast=auto_contrast)
        else:
            results = list(renderer.project_shots_iter(shots, RenderResultMode.ShotOnly, release_shots=release_shots,
                                                       mask=mask))
            img = results[0] if results else None
    finally:
        renderer.release()
    if img is None:
        return np.zeros((int(size[1]), int(size[0]), 4), dtype=np.uint8)
    out = np.array(img)                                   # copy: the buffer may be read-only
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.dstack([out, out, out, np.full_like(out, 255)])
    elif out.shape[2] == 3:
        out = np.dstack([out, np.full(out.shape[:2], 255, dtype=np.uint8)])
    return out


def erode_valid_mask(valid: ArrayLike, erosion_px: int) -> NDArray[np.bool_]:
    """Shrink a validity mask inward by ``erosion_px`` (4-neighbour erosions).

    Rendered footprints carry a 1-2 px antialiased rim where alpha is set but
    the colour has bled towards black; eroding it to nodata keeps that ring
    out of an orthomosaic's seams. Border pixels count as invalid.
    """
    m = np.asarray(valid, dtype=bool)
    if erosion_px <= 0 or not m.any():
        return m
    for _ in range(int(erosion_px)):
        p = np.pad(m, 1, constant_values=False)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def crop_to_content(image: ArrayLike, bounds: Bounds) -> Tuple[NDArray, Bounds]:
    """Trim fully transparent rows/columns and shrink the bounds to match."""
    img = np.asarray(image)
    alpha = img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else img
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return img, bounds
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    h, w = alpha.shape
    min_x, min_y, max_x, max_y = bounds
    sx, sy = (max_x - min_x) / w, (max_y - min_y) / h
    return img[r0:r1, c0:c1], (min_x + c0 * sx, max_y - r1 * sy, min_x + c1 * sx, max_y - r0 * sy)


def frame_orthophoto(ctx, mesh_data, texture_data, mesh, image: Union[str, ArrayLike], position: ArrayLike,
                     rotation: ArrayLike, fovy: float, width: int, height: int, polygon: Optional[ArrayLike] = None,
                     ground_resolution: Optional[float] = 0.1, fixed_size: Optional[Tuple[int, int]] = None,
                     translation_correction: ArrayLike = (0.0, 0.0, 0.0), rotation_correction: ArrayLike = (0.0, 0.0, 0.0),
                     bounds_correction_scale: float = 2.0, mask=None, edge_erosion_px: int = 2, max_dim: int = 8000,
                     convention: str = "drone_pose"):
    """One frame -> its orthophoto: the plugin's per-frame GeoTIFF recipe.

    The extent is where the frame's valid-area ``polygon`` (pixels; default
    the whole frame) lands on the terrain, cast through a camera whose
    rotation correction is scaled by ``bounds_correction_scale`` - 2.0 by
    default because the renderer applies the shot correction twice in
    effect (``CtxShot.get_correction``), and the bounds have to match what
    is drawn. Then the shot is rendered orthographically into that extent.

    :return: ``(image (H, W, 3) uint8, valid (H, W) bool, bounds)`` or
        ``None`` when fewer than three polygon vertices reach the terrain
    """
    from bambi.geo.camera import camera_from_pose
    from bambi.render.masks import default_mask_polygon, polygon_bounds, polygon_to_world

    aspect = width / height
    poly = default_mask_polygon(width, height) if polygon is None else np.asarray(polygon, dtype=np.float64)
    rc = np.asarray(rotation_correction, dtype=np.float64) * float(bounds_correction_scale)
    bounds_cam = camera_from_pose(position, rotation, fovy, aspect, translation_correction, rc)
    ground = polygon_to_world(poly, bounds_cam, width, height, mesh, legacy=True)
    bounds = polygon_bounds(ground)
    if bounds is None or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    size = render_size(bounds[2] - bounds[0], bounds[3] - bounds[1], ground_resolution, fixed_size, max_dim)
    shot = make_shot(ctx, image, position, rotation, fovy, aspect, translation_correction, rotation_correction,
                     lazy=False, convention=convention)
    rgba = render_orthographic(ctx, mesh_data, texture_data, shot, bounds, size, mask=mask, release_shots=True)
    valid = rgba[:, :, 3] > 0
    if edge_erosion_px > 0:
        valid = erode_valid_mask(valid, edge_erosion_px)
    return rgba[:, :, :3], valid, bounds


def render_integral_tiled(ctx, mesh_data, texture_data, shots: Sequence[object], bounds: Bounds,
                          size: Tuple[int, int], max_tile: int = 4096, camera_height: Optional[float] = None,
                          mask=None, auto_contrast: bool = True, crop: bool = False,
                          shot_footprints: Optional[Sequence[ArrayLike]] = None
                          ) -> Tuple[NDArray[np.uint8], Bounds]:
    """The light-field integral of many shots over a large extent, tile by tile.

    The plugin's ALFS step: the raster is cut into ``max_tile`` pixel tiles
    (:func:`tiles`), each rendered with its own orthographic sub-camera
    (:func:`tile_camera`) from only the shots whose footprint touches it
    (when ``shot_footprints`` - one ``(K, 2|3)`` ground polygon per shot -
    are given), and the tiles are assembled into one ``(H, W, 4)`` uint8
    RGBA image. ``crop=True`` trims fully transparent margins and returns
    the shrunk bounds.

    :return: ``(image, bounds)``
    """
    from alfspy.core.rendering import Resolution
    from alfspy.core.rendering.renderer import Renderer

    shots = list(shots)
    if camera_height is None:
        camera_height = float(np.asarray(mesh_data.vertices)[:, 2].max()) + 100.0
    w, h = int(size[0]), int(size[1])
    out = np.zeros((h, w, 4), dtype=np.uint8)
    global_cam = ortho_camera(bounds, camera_height)
    fp_bounds = None
    if shot_footprints is not None:
        fp_bounds = []
        for fp in shot_footprints:
            q = np.atleast_2d(np.asarray(fp, dtype=np.float64))[:, :2]
            q = q[np.isfinite(q).all(axis=1)]
            fp_bounds.append(None if len(q) == 0 else (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
    for tile in tiles((w, h), max_tile):
        tx, ty, tw, th = (int(v) for v in tile)
        tb = tile_bounds(bounds, (w, h), tile)
        if fp_bounds is None:
            tile_shots = shots
        else:
            tile_shots = [s for s, fb in zip(shots, fp_bounds)
                          if fb is not None and fb[0] <= tb[2] and fb[2] >= tb[0] and fb[1] <= tb[3] and fb[3] >= tb[1]]
        if not tile_shots:
            continue
        renderer = Renderer(Resolution(tw, th), ctx, tile_camera(global_cam, bounds, (w, h), tile), mesh_data,
                            texture_data)
        try:
            img = renderer.render_integral(tile_shots, mask=mask, save=False, release_shots=False,
                                           auto_contrast=auto_contrast)
        finally:
            renderer.release()
        if img is None:
            continue
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.dstack([arr, arr, arr, np.full_like(arr, 255)])
        elif arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, dtype=np.uint8)])
        out[ty:ty + th, tx:tx + tw] = arr[:th, :tw]
    if crop:
        return crop_to_content(out, bounds)
    return out, bounds
