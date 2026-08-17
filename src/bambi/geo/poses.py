# -*- coding: utf-8 -*-
"""Drone poses as arrays: geographic in, DEM-local out.

The pipeline works in a *DEM-local* metric frame - projected metres relative to
the DEM's origin - with the camera orientation as ``[tilt, roll, heading]``:
tilt from nadir (0 = straight down, 90 = horizon), roll (always 0 for a
gimbal), heading clockwise from north. The public BAMBI dataset, DJI's SRT
track and AirData logs all express poses *geographically*: WGS84
latitude/longitude, an altitude, and gimbal pitch/roll/yaw. This module is the
conversion between the two, and nothing else - no files, no per-frame records.

Every function takes and returns numpy arrays (or a small frozen dataclass of
arrays), so the same code serves the pipeline, a notebook, or a test that
builds a pose in one line. Reading a poses file into these arrays and writing
them back out belongs to the io/cli edge (:mod:`bambi.io.poses`).

Conventions, pinned by tests:

* ``lla`` is ``(N, 3)`` ``[lat, lon, alt]`` in degrees / metres.
* ``pry`` is ``(N, 3)`` ``[gimbal_pitch, roll, heading]`` in degrees, DJI-style
  (pitch -90 = nadir).
* ``positions`` is ``(N, 3)`` DEM-local metres ``[east, north, up]``.
* ``rotations`` is ``(N, 3)`` ``[tilt, roll, heading]`` in degrees, i.e. exactly
  the ``rotation`` entry of a poses file - and the input
  ``alfspy.core.util.pyrrs.quaternion_from_drone_pose`` expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Origin",
    "Poses",
    "make_origin",
    "geographic_to_local",
    "local_to_geographic",
    "gimbal_to_rotation",
    "rotation_to_gimbal",
    "poses_from_geographic",
    "poses_to_geographic",
    "grid_convergence",
]


@dataclass(frozen=True)
class Origin:
    """The DEM-local frame: a WGS84 anchor and the projected CRS it maps to.

    ``easting``/``northing`` are the anchor's projected coordinates - cached
    so repeated conversions do not re-project the anchor - and ``altitude`` is
    subtracted from every pose altitude to give DEM-local ``up``.
    """
    latitude: float
    longitude: float
    altitude: float
    epsg: int
    easting: float
    northing: float

    def as_dict(self) -> dict:
        """The ``origin`` block a poses file carries."""
        return {"latitude": self.latitude, "longitude": self.longitude,
                "altitude": self.altitude}


@dataclass(frozen=True)
class Poses:
    """A flight's poses in the DEM-local frame.

    :ivar positions: ``(N, 3)`` ``[east, north, up]`` metres from the origin
    :ivar rotations: ``(N, 3)`` ``[tilt, roll, heading]`` degrees
    :ivar origin: the frame they are expressed in
    """
    positions: NDArray[np.float64]
    rotations: NDArray[np.float64]
    origin: Origin

    def __len__(self) -> int:
        return int(self.positions.shape[0])


# ---------------------------------------------------------------------------
# origin
# ---------------------------------------------------------------------------

def make_origin(latitude: float, longitude: float, altitude: float, epsg: int) -> Origin:
    """Anchor a DEM-local frame at a WGS84 point.

    :param latitude: anchor latitude (deg)
    :param longitude: anchor longitude (deg)
    :param altitude: anchor altitude (m); becomes DEM-local ``up = 0``
    :param epsg: projected CRS the local axes follow (e.g. 32633 for UTM 33N)
    """
    east, north = _project(np.array([latitude], float), np.array([longitude], float), epsg)
    return Origin(float(latitude), float(longitude), float(altitude), int(epsg),
                  float(east[0]), float(north[0]))


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------

def geographic_to_local(lla: ArrayLike, origin: Origin) -> NDArray[np.float64]:
    """WGS84 ``[lat, lon, alt]`` -> DEM-local ``[east, north, up]``.

    Mirrors the pipeline's pose extractors exactly: project lat/lon into the
    origin's CRS, subtract the projected origin, subtract the origin altitude.

    :param lla: ``(N, 3)`` or ``(3,)``
    :return: ``(N, 3)`` float64
    """
    lla = np.atleast_2d(np.asarray(lla, dtype=np.float64))
    if lla.shape[1] != 3:
        raise ValueError(f"lla must be (N, 3) [lat, lon, alt], got {lla.shape}")
    east, north = _project(lla[:, 0], lla[:, 1], origin.epsg)
    out = np.empty_like(lla)
    out[:, 0] = east - origin.easting
    out[:, 1] = north - origin.northing
    out[:, 2] = lla[:, 2] - origin.altitude
    return out


def local_to_geographic(positions: ArrayLike, origin: Origin) -> NDArray[np.float64]:
    """Inverse of :func:`geographic_to_local`. ``(N, 3)`` -> ``(N, 3)`` ``[lat, lon, alt]``."""
    p = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if p.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3) [east, north, up], got {p.shape}")
    lat, lon = _unproject(p[:, 0] + origin.easting, p[:, 1] + origin.northing, origin.epsg)
    out = np.empty_like(p)
    out[:, 0] = lat
    out[:, 1] = lon
    out[:, 2] = p[:, 2] + origin.altitude
    return out


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------

def gimbal_to_rotation(pry: ArrayLike, heading_offset: ArrayLike = 0.0) -> NDArray[np.float64]:
    """DJI gimbal ``[pitch, roll, yaw]`` -> pose ``[tilt, roll, heading]``.

    ``tilt = (pitch + 90) mod 360``: DJI reports -90 for straight down and 0
    for the horizon; the pose convention is 0 for straight down. Roll is
    passed through (0 for a gimbal). Heading is the gimbal yaw plus an
    optional per-pose offset (see :func:`grid_convergence`), wrapped to
    ``[0, 360)``.

    :param pry: ``(N, 3)`` or ``(3,)`` degrees
    :param heading_offset: scalar or ``(N,)`` degrees added to the heading
    :return: ``(N, 3)`` float64
    """
    pry = np.atleast_2d(np.asarray(pry, dtype=np.float64))
    if pry.shape[1] != 3:
        raise ValueError(f"pry must be (N, 3) [pitch, roll, yaw], got {pry.shape}")
    out = np.empty_like(pry)
    out[:, 0] = np.mod(pry[:, 0] + 90.0, 360.0)
    out[:, 1] = pry[:, 1]
    out[:, 2] = np.mod(pry[:, 2] + np.asarray(heading_offset, dtype=np.float64), 360.0)
    return out


def rotation_to_gimbal(rotations: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`gimbal_to_rotation`: ``[tilt, roll, heading]`` -> ``[pitch, roll, yaw]``.

    Pitch is returned in DJI's ``(-180, 180]`` style so nadir reads -90 again.
    """
    r = np.atleast_2d(np.asarray(rotations, dtype=np.float64))
    if r.shape[1] != 3:
        raise ValueError(f"rotations must be (N, 3) [tilt, roll, heading], got {r.shape}")
    out = np.empty_like(r)
    pitch = np.mod(r[:, 0] - 90.0 + 180.0, 360.0) - 180.0
    out[:, 0] = pitch
    out[:, 1] = r[:, 1]
    out[:, 2] = np.mod(r[:, 2], 360.0)
    return out


def grid_convergence(lla: ArrayLike, epsg: int) -> NDArray[np.float64]:
    """Angle from true north to projected-grid north at each point, degrees.

    A gimbal heading is measured against true north; the DEM-local axes follow
    the projected CRS, whose grid north diverges from true north away from the
    central meridian (about +-2 deg across a UTM zone). Add this to a heading to
    express it in the local frame. Vectorised over ``N``.

    The pose extractors carry the same correction behind ``apply_correction``,
    off by default - and with a typo'd source CRS (``EPSG:4236``) that would
    have raised had it ever been enabled. This is the working version.
    """
    from pyproj import Geod

    lla = np.atleast_2d(np.asarray(lla, dtype=np.float64))
    lat, lon = lla[:, 0], lla[:, 1]
    geod = Geod(ellps="WGS84")
    # A point 1 m due (true) north of each pose ...
    n_lon, n_lat, _ = geod.fwd(lon, lat, np.zeros_like(lat), np.ones_like(lat))
    e0, n0 = _project(lat, lon, epsg)
    e1, n1 = _project(np.asarray(n_lat), np.asarray(n_lon), epsg)
    # ... and the grid bearing of that true-north step - which IS the grid
    # heading of a camera facing true north, so ``heading + convergence`` is the
    # DEM-local heading. East of the central meridian this is negative (a
    # true-north step lands slightly west of grid north). Note pyproj's
    # ``meridian_convergence`` uses the opposite sign (grid -> true); the
    # end-to-end pointing test in tests/test_geo_poses.py pins this one.
    return np.degrees(np.arctan2(e1 - e0, n1 - n0))


# ---------------------------------------------------------------------------
# both together
# ---------------------------------------------------------------------------

def poses_from_geographic(lla: ArrayLike, pry: ArrayLike, origin: Origin,
                          apply_grid_convergence: bool = False) -> Poses:
    """Build DEM-local :class:`Poses` from geographic positions and gimbal angles.

    :param lla: ``(N, 3)`` ``[lat, lon, alt]``
    :param pry: ``(N, 3)`` ``[gimbal_pitch, roll, yaw]`` degrees
    :param origin: the DEM-local frame
    :param apply_grid_convergence: rotate headings from true north to grid
        north (off by default, matching the pipeline's extractors)
    """
    lla = np.atleast_2d(np.asarray(lla, dtype=np.float64))
    pry = np.atleast_2d(np.asarray(pry, dtype=np.float64))
    if lla.shape[0] != pry.shape[0]:
        raise ValueError(f"lla has {lla.shape[0]} rows, pry has {pry.shape[0]}")
    offset = grid_convergence(lla, origin.epsg) if apply_grid_convergence else 0.0
    return Poses(positions=geographic_to_local(lla, origin),
                 rotations=gimbal_to_rotation(pry, offset),
                 origin=origin)


def poses_to_geographic(poses: Poses) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of :func:`poses_from_geographic`: ``(lla, pry)`` arrays."""
    return local_to_geographic(poses.positions, poses.origin), rotation_to_gimbal(poses.rotations)


# ---------------------------------------------------------------------------
# projection helpers (pyproj kept out of the public signatures)
# ---------------------------------------------------------------------------

_TRANSFORMERS: dict = {}


def _transformer(epsg: int, inverse: bool = False):
    from pyproj import CRS, Transformer

    key = (int(epsg), inverse)
    if key not in _TRANSFORMERS:
        src, dst = CRS.from_epsg(4326), CRS.from_epsg(int(epsg))
        if inverse:
            src, dst = dst, src
        # always_xy=False on purpose: the extractors call
        # transformer.transform(lat, lon) with the CRS's own axis order.
        _TRANSFORMERS[key] = Transformer.from_crs(src, dst)
    return _TRANSFORMERS[key]


def _project(lat: NDArray, lon: NDArray, epsg: int) -> Tuple[NDArray, NDArray]:
    east, north = _transformer(epsg).transform(lat, lon)
    return np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)


def _unproject(east: NDArray, north: NDArray, epsg: int) -> Tuple[NDArray, NDArray]:
    lat, lon = _transformer(epsg, inverse=True).transform(east, north)
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
