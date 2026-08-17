# -*- coding: utf-8 -*-
"""A synthetic survey scene: rolling terrain, markers at known coordinates,
and cameras looking at them from any tilt and heading.

Why this exists: every flight in the public BAMBI dataset is nadir, and the
one oblique flight we have seen is neither ours nor publishable. Whether the
pointing maths is right at 45 or 80 degrees of tilt therefore cannot be shown
on data - but it can be shown exactly here. The scene is procedural, so the
truth is analytic: a marker *is* at ``markers[k]``; project it into a camera,
cast the pixel back onto the terrain, and the two must agree.

The layout matches the pipeline (mesh from :func:`bambi.geo.dem.heightfield_mesh`,
DEM-local metres, poses ``[tilt, roll, heading]``), so anything proven here
transfers to real flights unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bambi.geo.dem import heightfield_mesh, sample_heightfield
from bambi.geo.poses import Poses, make_origin

__all__ = ["SyntheticScene", "make_terrain", "make_scene", "look_at_poses", "forward_vector",
           "visible_markers"]


@dataclass(frozen=True)
class SyntheticScene:
    """Terrain + markers + poses with exact ground truth.

    All coordinates are DEM-local metres (``[east, north, up]``); the mesh
    origin is the grid's south-west corner at the terrain's lowest point.
    """
    elevation: NDArray[np.float64]      # (H, W) metres above the base
    cell_size: Tuple[float, float]
    vertices: NDArray[np.float32]       # (H*W, 3)
    faces: NDArray[np.uint32]           # (M, 3)
    markers: NDArray[np.float64]        # (K, 3) on the terrain surface
    poses: Poses                        # one per view
    fovy: float
    frame_size: Tuple[int, int]         # (width, height)
    aim: NDArray[np.float64]            # (3,) the point every camera looks at

    @property
    def mesh(self):
        import trimesh
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

    @property
    def extent(self) -> Tuple[float, float]:
        rows, cols = self.elevation.shape
        return ((cols - 1) * self.cell_size[0], (rows - 1) * self.cell_size[1])


def make_terrain(shape: Tuple[int, int] = (81, 81), cell_size: float = 2.5,
                 relief: float = 12.0, seed: int = 7) -> NDArray[np.float64]:
    """Smooth rolling terrain: a few broad sine hills plus gentle noise.

    :return: ``(H, W)`` elevation in metres, minimum 0
    """
    rows, cols = shape
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    x = x * cell_size / ((cols - 1) * cell_size)
    y = y * cell_size / ((rows - 1) * cell_size)
    z = (np.sin(2.3 * np.pi * x + 0.4) * np.cos(1.7 * np.pi * y - 0.9)
         + 0.5 * np.sin(4.1 * np.pi * (x + y)) + 0.35 * np.cos(5.3 * np.pi * (x - 0.5 * y)))
    z = z + 0.15 * rng.standard_normal(z.shape)
    # smooth the noise a little so the mesh stays well behaved
    k = np.array([1, 4, 6, 4, 1], float); k = np.outer(k, k); k /= k.sum()
    zp = np.pad(z, 2, mode="edge")
    sm = np.zeros_like(z)
    for i in range(5):
        for j in range(5):
            sm += k[i, j] * zp[i:i + rows, j:j + cols]
    sm -= sm.min()
    return sm / sm.max() * relief


def forward_vector(tilt_deg: ArrayLike, heading_deg: ArrayLike) -> NDArray[np.float64]:
    """ENU direction a gimbal points at ``tilt`` off nadir on ``heading``."""
    t = np.deg2rad(np.asarray(tilt_deg, float))
    h = np.deg2rad(np.asarray(heading_deg, float))
    return np.stack([np.sin(t) * np.sin(h), np.sin(t) * np.cos(h), -np.cos(t)], axis=-1)


def look_at_poses(aim: ArrayLike, tilts_deg: ArrayLike, headings_deg: ArrayLike,
                  height_above_aim: float, roll_deg: float = 0.0,
                  origin=None) -> Poses:
    """Cameras whose optical axis passes through ``aim`` at the given tilt/heading.

    For each ``(tilt, heading)`` pair the camera sits ``height_above_aim`` metres
    above the aim point, displaced *backwards* along the view direction so
    the aim point is the frame centre. Nadir puts it straight above.

    :param aim: ``(3,)`` DEM-local point in the frame centre
    :param tilts_deg: ``(N,)`` tilt off nadir
    :param headings_deg: ``(N,)`` heading clockwise from north (same length)
    :param height_above_aim: vertical distance camera - aim, metres
    """
    aim = np.asarray(aim, float).reshape(3)
    tilts = np.atleast_1d(np.asarray(tilts_deg, float))
    heads = np.atleast_1d(np.asarray(headings_deg, float))
    if tilts.shape != heads.shape:
        raise ValueError("tilts_deg and headings_deg must have the same length")
    if np.any(tilts >= 90.0):
        raise ValueError("a camera at or above the horizon cannot look at a ground point")
    fwd = forward_vector(tilts, heads)                    # (N, 3), z component = -cos(tilt) < 0
    dist = float(height_above_aim) / np.cos(np.deg2rad(tilts))
    positions = aim[None, :] - fwd * dist[:, None]
    rotations = np.column_stack([tilts, np.full_like(tilts, roll_deg), np.mod(heads, 360.0)])
    if origin is None:
        origin = make_origin(46.0, 14.0, 500.0, 32633)
    return Poses(positions=positions, rotations=rotations, origin=origin)


def visible_markers(scene: "SyntheticScene", pose_index: int, mesh=None,
                    tolerance: float = 0.05) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Which markers a camera actually sees: inside the frame *and* not hidden
    behind terrain (at 80 degrees of tilt a hill in front of a marker is common).

    :return: ``visible (K,)`` mask and the ``(K, 2)`` pixel coordinates (NaN
        where the marker is behind the camera)
    """
    from bambi.geo.camera import camera_from_pose, world_to_pixel

    w, h = scene.frame_size
    pos, rot = scene.poses.positions[pose_index], scene.poses.rotations[pose_index]
    camera = camera_from_pose(pos, rot, scene.fovy, w / h)
    px = world_to_pixel(scene.markers, camera, w, h)
    inside = np.all(np.isfinite(px), axis=1)
    inside &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    mesh = scene.mesh if mesh is None else mesh
    origins = np.repeat(pos[None, :], inside.sum(), axis=0)
    dirs = scene.markers[inside] - pos
    dist = np.linalg.norm(dirs, axis=1)
    dirs = dirs / dist[:, None]
    # first hit along the camera->marker ray; occluded when it stops short of the marker
    loc, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    first = np.full(inside.sum(), np.inf)
    if len(loc):
        d = np.linalg.norm(loc - origins[ray_idx], axis=1)
        np.minimum.at(first, ray_idx, d)
    unoccluded = first >= dist - tolerance
    visible = inside.copy()
    visible[np.flatnonzero(inside)[~unoccluded]] = False
    return visible, px


def make_scene(tilts_deg: ArrayLike = (0.0, 45.0, 80.0),
               headings_deg: ArrayLike = (0.0, 120.0, 250.0),
               n_markers: int = 12, height_above_aim: float = 60.0, fovy: float = 50.0,
               frame_size: Tuple[int, int] = (1024, 1024), marker_spread: float = 18.0,
               shape: Tuple[int, int] = (81, 81), cell_size: float = 2.5,
               relief: float = 12.0, seed: int = 7) -> SyntheticScene:
    """The default oblique-view scene: every tilt at every heading, markers
    scattered on the terrain around the aim point.

    :param tilts_deg: tilts to include (each combined with every heading)
    :param headings_deg: headings to include
    :param n_markers: markers placed on the surface within ``marker_spread`` of the aim
    :param height_above_aim: camera height above the aim point, metres
    """
    elevation = make_terrain(shape, cell_size, relief, seed)
    vertices, faces, _ = heightfield_mesh(elevation, (cell_size, cell_size), base_elevation=0.0)
    ex, ey = (shape[1] - 1) * cell_size, (shape[0] - 1) * cell_size
    centre = np.array([ex / 2.0, ey / 2.0])
    aim_z = sample_heightfield(elevation, (cell_size, cell_size), centre[None, :], base_elevation=0.0)[0]
    aim = np.array([centre[0], centre[1], aim_z])

    rng = np.random.default_rng(seed + 1)
    xy = centre[None, :] + rng.uniform(-marker_spread, marker_spread, size=(n_markers, 2))
    z = sample_heightfield(elevation, (cell_size, cell_size), xy, base_elevation=0.0)
    markers = np.column_stack([xy, z])

    tg, hg = np.meshgrid(np.atleast_1d(tilts_deg), np.atleast_1d(headings_deg), indexing="ij")
    poses = look_at_poses(aim, tg.ravel(), hg.ravel(), height_above_aim)
    return SyntheticScene(elevation=elevation, cell_size=(cell_size, cell_size), vertices=vertices,
                          faces=faces, markers=markers, poses=poses, fovy=float(fovy),
                          frame_size=tuple(int(v) for v in frame_size), aim=aim)
