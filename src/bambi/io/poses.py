# -*- coding: utf-8 -*-
"""Read and write poses files as arrays.

Two schemas exist in the wild and both are handled:

* **pipeline** (``poses.json`` / ``poses_t.json`` written by the extractors and
  consumed by the QGIS plugin): each image carries DEM-local ``location`` and
  ``rotation`` (plus ``fovy``, ``lat``/``lng``, ``timestamp``, ``imagefile``),
  and the file carries an ``origin`` block.
* **public dataset** (``<flight>_matched_poses.json`` from bambi-eco/Dataset):
  each image carries geographic ``lat``/``lng``/``alt`` and ``pitch``/``roll``/
  ``yaw``, plus an ``origin`` block. There is no ``location``; converting to
  DEM-local needs an origin - see :func:`bambi.geo.poses.geographic_to_local`.

  **The public ``pitch`` is not DJI gimbal pitch.** It is already the pose
  convention - tilt from nadir - so a nadir flight reads ~0/360, not -90; the
  Dataset repo's ``add_relative_dem_position_to_poses.py`` copies
  ``[pitch, roll, yaw]`` into ``rotation`` verbatim. This reader therefore
  exposes those angles as ``rotations`` directly and does *not* add 90 again.
  DJI SRT/AirData sources (nadir = -90) go through
  :func:`bambi.geo.poses.gimbal_to_rotation` instead - that path is used by the
  extractors, not by this reader.

Reading yields plain arrays (and the per-image metadata as parallel lists);
writing takes arrays. The compute code never touches this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from bambi.geo.poses import (Origin, Poses, geographic_to_local, grid_convergence,
                             make_origin)

PathLike = Union[str, Path]


@dataclass
class PosesFile:
    """A poses file, split into arrays plus everything that is not array-shaped."""
    #: which schema the file used: ``"pipeline"`` or ``"geographic"``
    schema: str
    #: ``(N, 3)`` DEM-local ``[east, north, up]`` - only for the pipeline schema
    positions: Optional[NDArray[np.float64]]
    #: ``(N, 3)`` ``[tilt, roll, heading]`` in the pose convention. Present for
    #: BOTH schemas: the pipeline schema stores it as ``rotation``, the public
    #: geographic schema stores the same convention as ``pitch``/``roll``/``yaw``.
    rotations: Optional[NDArray[np.float64]]
    #: ``(N, 3)`` ``[lat, lon, alt]`` when the file carries it (both schemas usually do)
    lla: Optional[NDArray[np.float64]]
    #: ``(N,)`` vertical field of view, degrees (NaN where absent)
    fovy: NDArray[np.float64]
    #: per-image file names (``""`` where absent)
    imagefiles: List[str]
    #: per-image ISO timestamps (``""`` where absent)
    timestamps: List[str]
    #: the ``origin`` block, or ``None``
    origin: Optional[Dict[str, float]]
    #: every top-level key that is not ``images`` (drone, camera, mask, ...)
    header: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.imagefiles)


def read_poses(path: PathLike) -> PosesFile:
    """Load a poses file of either schema into arrays.

    :param path: the JSON file
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    images = data.get("images", [])
    n = len(images)
    header = {k: v for k, v in data.items() if k != "images"}
    origin = data.get("origin")

    def col(key: str, width: int = 3):
        vals = [im.get(key) for im in images]
        if all(v is None for v in vals):
            return None
        out = np.full((n, width), np.nan, dtype=np.float64)
        for i, v in enumerate(vals):
            if v is not None:
                out[i, :len(v)] = np.asarray(v, dtype=np.float64)[:width]
        return out

    positions = col("location")
    rotations = col("rotation")

    lla = None
    if images and "lat" in images[0]:
        alt_key = "alt" if "alt" in images[0] else None
        lla = np.full((n, 3), np.nan)
        for i, im in enumerate(images):
            lla[i, 0] = im.get("lat", np.nan)
            lla[i, 1] = im.get("lng", im.get("lon", np.nan))
            if alt_key:
                lla[i, 2] = im.get(alt_key, np.nan)

    if rotations is None and images and "pitch" in images[0]:
        # Public-dataset schema: pitch/roll/yaw already ARE the pose
        # convention (see module docstring); wrap tilt/heading into [0, 360).
        rotations = np.column_stack([
            np.mod([im.get("pitch", np.nan) for im in images], 360.0),
            [im.get("roll", 0.0) for im in images],
            np.mod([im.get("yaw", np.nan) for im in images], 360.0),
        ]).astype(np.float64)

    fovy = np.full(n, np.nan)
    for i, im in enumerate(images):
        f = im.get("fovy")
        if isinstance(f, (list, tuple)):
            f = f[0] if f else None
        if f is not None:
            fovy[i] = float(f)

    schema = "pipeline" if positions is not None else "geographic"
    return PosesFile(
        schema=schema, positions=positions, rotations=rotations, lla=lla,
        fovy=fovy,
        imagefiles=[str(im.get("imagefile", "")) for im in images],
        timestamps=[str(im.get("timestamp", "")) for im in images],
        origin=origin, header=header,
    )


def to_local_poses(pf: PosesFile, epsg: int, origin: Optional[Origin] = None,
                   apply_grid_convergence: bool = False) -> Poses:
    """DEM-local :class:`Poses` from a loaded file of either schema.

    A pipeline-schema file is already local and is returned as is (its
    ``origin`` block becomes the :class:`Origin`). A geographic-schema file is
    converted through :func:`bambi.geo.poses.poses_from_geographic`.

    :param pf: a loaded poses file
    :param epsg: projected CRS for the local frame (must match the DEM)
    :param origin: anchor to use; defaults to the file's ``origin`` block
    """
    if origin is None:
        if pf.origin is None:
            raise ValueError("poses file has no origin block; pass origin=")
        origin = make_origin(pf.origin["latitude"], pf.origin["longitude"],
                             float(pf.origin.get("altitude", 0.0)), epsg)
    if pf.schema == "pipeline":
        return Poses(positions=pf.positions, rotations=pf.rotations, origin=origin)
    if pf.lla is None or pf.rotations is None:
        raise ValueError("geographic poses file lacks lat/lng/alt or pitch/roll/yaw")
    rotations = pf.rotations
    if apply_grid_convergence:
        rotations = rotations.copy()
        rotations[:, 2] = np.mod(rotations[:, 2] + grid_convergence(pf.lla, origin.epsg), 360.0)
    return Poses(positions=geographic_to_local(pf.lla, origin), rotations=rotations, origin=origin)


def write_poses(path: PathLike, poses: Poses, imagefiles: List[str],
                fovy: Optional[NDArray] = None, lla: Optional[NDArray] = None,
                timestamps: Optional[List[str]] = None,
                header: Optional[Dict[str, Any]] = None) -> None:
    """Write DEM-local poses in the pipeline schema.

    :param path: destination JSON
    :param poses: DEM-local poses
    :param imagefiles: one file name per pose
    :param fovy: ``(N,)`` vertical FOV in degrees, written as ``[fovy]`` like the extractors
    :param lla: ``(N, 3)`` ``[lat, lon, alt]``; written as ``lat``/``lng`` when given
    :param timestamps: per-pose ISO timestamps
    :param header: extra top-level keys (drone, camera, mask, samplingRate)
    """
    n = len(poses)
    if len(imagefiles) != n:
        raise ValueError(f"{len(imagefiles)} imagefiles for {n} poses")
    images = []
    for i in range(n):
        im: Dict[str, Any] = {
            "imagefile": imagefiles[i],
            "location": [float(v) for v in poses.positions[i]],
            "rotation": [float(v) for v in poses.rotations[i]],
        }
        if fovy is not None and np.isfinite(fovy[i]):
            im["fovy"] = [float(fovy[i])]
        if lla is not None:
            im["lat"] = float(lla[i, 0])
            im["lng"] = float(lla[i, 1])
        if timestamps is not None and timestamps[i]:
            im["timestamp"] = timestamps[i]
        images.append(im)
    out: Dict[str, Any] = dict(header or {})
    out["origin"] = poses.origin.as_dict()
    out["images"] = images
    Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")


def epochs_from_timestamps(timestamps) -> "np.ndarray":
    """ISO-8601 timestamps (with timezone, as the extractors write them) ->
    ``(N,)`` epoch seconds, NaN where missing or unparseable.

    Thermal and RGB frames share this clock (it is the SRT capture time), so
    it is what :func:`bambi.tracking.matching.match_frames_by_time` pairs on.
    """
    from datetime import datetime

    out = np.full(len(timestamps), np.nan)
    for i, ts in enumerate(timestamps):
        if not ts:
            continue
        try:
            out[i] = datetime.fromisoformat(str(ts)).timestamp()
        except (ValueError, TypeError):
            pass
    return out
