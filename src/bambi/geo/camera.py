# -*- coding: utf-8 -*-
"""Pose arrays -> alfspy cameras, correctly, on either alfspy backend.

This is the single place a DEM-local pose becomes a camera the renderer and
the ray-caster agree on. Two things have to line up and both have bitten:

* **The pose rotation.** ``[tilt, roll, heading]`` goes through alfspy's
  ``quaternion_from_drone_pose``, which applies the heading about world up
  *after* the tilt. The older ``quaternion_from_eulers(e, 'zyx')`` spelling
  applied it about the camera's own optical axis - exact at nadir, up to 128
  degrees wrong at the horizon.
* **The ray convention of the installed alfspy.** Older builds turn camera
  rays into world rays with ``R33.T``, newer ones with ``R33``. Both are in
  use. :func:`ray_convention` probes once (one ray through a sphere centred on
  the camera) and :func:`cameras_from_poses` conjugates the quaternion for the
  legacy build, so the same world ray comes out either way.

The **1x correction rule** is kept: a per-pose correction is subtracted from
the pose eulers exactly once, matching what alfspy's own renderer applies via
``CtxShot.get_correction()``, so geo-referencing and rendered output stay
consistent (see the QGIS plugin's GeoTIFF rotation-sign incident).

Numeric detail preserved on purpose: corrections are held in **float32**,
exactly as the plugin's ``Vector3(..., dtype='f4')`` did, so results are
bit-comparable with the parity oracle. It is ~1e-7 rad and irrelevant in
practice; it is kept so parity means parity.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bambi.geo.poses import Poses

__all__ = ["ray_convention", "camera_from_pose", "cameras_from_poses", "world_to_pixel"]

_RAY_CONVENTION: Optional[str] = None


def ray_convention(force_probe: bool = False) -> str:
    """Detect how the installed alfspy rotates camera-space rays into the world.

    :param force_probe: re-run the probe instead of using the cached answer
    :return: ``"legacy"`` (rays built with ``R33.T``) or ``"fixed"`` (``R33``)
    :raises RuntimeError: when alfspy/pyrr/trimesh are missing, or the probe
        cannot tell the two apart
    """
    global _RAY_CONVENTION
    if _RAY_CONVENTION is not None and not force_probe:
        return _RAY_CONVENTION

    try:
        import trimesh
        from pyrr import Quaternion, Vector3
        from alfspy.core.convert.convert import pixel_to_world_coord
        from alfspy.core.rendering import Camera
    except ImportError as exc:
        raise RuntimeError(f"alfspy / pyrr / trimesh not available: {exc}") from exc

    # Deliberately asymmetric so R33 and R33.T give clearly different rays.
    eulers = np.deg2rad([35.0, 12.0, 70.0])
    camera = Camera(fovy=50.0, aspect_ratio=1.0, position=Vector3([0.0, 0.0, 0.0]),
                    rotation=Quaternion.from_eulers(Vector3(eulers)))
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=100.0)
    hits = pixel_to_world_coord([256], [128], 512, 512, sphere, camera, include_misses=False)
    hits = np.reshape(np.asarray(hits, dtype=float), (-1, 3))
    if len(hits) != 1 or not np.all(np.isfinite(hits)):
        raise RuntimeError("Could not probe the alfspy ray convention: the test ray missed the sphere.")
    hit = hits[0] / np.linalg.norm(hits[0])

    rot = np.asarray(camera.transform.rotation.matrix33, dtype=np.float64)
    t = np.tan(np.deg2rad(50.0) / 2.0)
    local = np.array([0.0 * t, 0.5 * t, -1.0])
    scores = {"fixed": float(hit @ (local @ rot) / np.linalg.norm(local @ rot)),
              "legacy": float(hit @ (local @ rot.T) / np.linalg.norm(local @ rot.T))}
    best = max(scores, key=scores.get)
    if scores[best] < 0.999 or sorted(scores.values())[-2] > 0.99:
        raise RuntimeError(f"Could not probe the alfspy ray convention unambiguously (scores: {scores}).")
    _RAY_CONVENTION = best
    return best


def camera_from_pose(position: ArrayLike, rotation: ArrayLike, fovy: float,
                     aspect_ratio: float = 1.0,
                     translation_correction: ArrayLike = (0.0, 0.0, 0.0),
                     rotation_correction: ArrayLike = (0.0, 0.0, 0.0)):
    """One alfspy ``Camera`` from one DEM-local pose.

    :param position: ``(3,)`` DEM-local ``[east, north, up]`` metres
    :param rotation: ``(3,)`` ``[tilt, roll, heading]`` degrees
    :param fovy: vertical field of view, degrees
    :param aspect_ratio: width / height of the frames the camera saw
    :param translation_correction: ``(3,)`` metres, added to the position
    :param rotation_correction: ``(3,)`` **radians**, subtracted once from the
        pose eulers (the 1x rule)
    """
    try:
        from pyrr import Vector3
        from alfspy.core.rendering import Camera
        from alfspy.core.util.pyrrs import quaternion_from_drone_pose
    except ImportError as exc:
        raise RuntimeError(f"alfspy / pyrr not available: {exc}") from exc

    pos = Vector3(np.asarray(position, dtype=np.float64)) + Vector3(
        np.asarray(translation_correction, dtype=np.float32))
    eulers = Vector3([np.deg2rad(float(v) % 360.0) for v in np.asarray(rotation, dtype=np.float64)[:3]]) \
        - Vector3(np.asarray(rotation_correction, dtype=np.float32))
    quat = quaternion_from_drone_pose(np.degrees(eulers))
    if ray_convention() == "legacy":
        quat = quat.conjugate
    return Camera(fovy=float(fovy), aspect_ratio=float(aspect_ratio), position=pos, rotation=quat)


def cameras_from_poses(poses: Poses, fovy: ArrayLike, aspect_ratio: float = 1.0,
                       translation_corrections: Optional[ArrayLike] = None,
                       rotation_corrections: Optional[ArrayLike] = None,
                       indices: Optional[ArrayLike] = None) -> List[object]:
    """alfspy ``Camera`` objects for (a subset of) poses.

    :param poses: DEM-local poses
    :param fovy: scalar or ``(N,)`` vertical FOV in degrees
    :param aspect_ratio: width / height of the frames
    :param translation_corrections: ``(N, 3)`` metres or ``None``
    :param rotation_corrections: ``(N, 3)`` radians or ``None``
    :param indices: which poses; default all
    :return: list of cameras, one per selected index, in that order
    """
    n = len(poses)
    idx = np.arange(n) if indices is None else np.asarray(indices, dtype=int)
    fov = np.broadcast_to(np.asarray(fovy, dtype=np.float64), (n,))
    tc = np.zeros((n, 3)) if translation_corrections is None else np.asarray(translation_corrections, float)
    rc = np.zeros((n, 3)) if rotation_corrections is None else np.asarray(rotation_corrections, float)
    if tc.shape != (n, 3) or rc.shape != (n, 3):
        raise ValueError(f"corrections must be ({n}, 3)")
    return [camera_from_pose(poses.positions[i], poses.rotations[i], fov[i], aspect_ratio, tc[i], rc[i])
            for i in idx]


def world_to_pixel(points: ArrayLike, camera, width: int, height: int) -> NDArray[np.float64]:
    """Project world points through an alfspy camera into pixel coordinates.

    ``(N, 3)`` DEM-local points -> ``(N, 2)`` ``[x, y]`` pixels, top-left
    origin. Points behind the camera get NaN. Reimplements
    ``alfspy.core.convert.world_to_pixel_coord`` in float64 with a correct
    ``(N, 4)`` perspective divide (the upstream helper broadcast wrongly for
    more than one point in older builds).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    homo = np.ones((len(pts), 4)); homo[:, :3] = pts
    view = np.asarray(camera.get_view(), dtype=np.float64)
    proj = np.asarray(camera.get_proj(), dtype=np.float64)
    ndc = (homo @ view) @ proj
    w = ndc[:, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = ndc / w
    out = np.empty((len(pts), 2))
    out[:, 0] = (ndc[:, 0] + 1.0) * width / 2.0
    out[:, 1] = height - (ndc[:, 1] + 1.0) * height / 2.0
    out[w[:, 0] <= 0] = np.nan
    return out
