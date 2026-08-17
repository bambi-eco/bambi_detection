# -*- coding: utf-8 -*-
"""The file edge for pose corrections: ``correction.json`` in its four dialects.

A correction is a small translation (metres) and rotation (**radians**, about
x/y/z) applied to every pose, optionally overridden for frame ranges. The
same idea has been written four ways:

* plugin ``correction.json``:      ``additional: [{start, end, translation, rotation}]``
* plugin pipeline config:          ``additional_corrections: [{start, end, ...}]``
* alfspy corrections file:         ``default: {...}, corrections: [{"start frame", "end frame", ...}]``
* public dataset ``<id>_correction.json``: ``fine_corrections: [{start_frame, end_frame, ...}]``

All four resolve the same way - a frame inside a range takes that range's
values, everything else takes the defaults - and all four come out of here as
two ``(N, 3)`` arrays ready for :func:`bambi.geo.camera.cameras_from_poses`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, Path]

__all__ = ["Corrections", "read_corrections", "parse_corrections", "corrections_for_frames"]

_ZERO = {"x": 0.0, "y": 0.0, "z": 0.0}


def _xyz(d: Any) -> Tuple[float, float, float]:
    if d is None:
        return (0.0, 0.0, 0.0)
    if isinstance(d, dict):
        return (float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0)))
    v = [float(x) for x in d]
    return (v[0], v[1], v[2])


@dataclass(frozen=True)
class Corrections:
    """Resolved correction: defaults plus ``(start, end, translation, rotation)`` ranges (inclusive)."""
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)          # radians
    ranges: List[Tuple[int, int, Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=list)

    @property
    def is_identity(self) -> bool:
        return (not any(self.translation) and not any(self.rotation)
                and all(not any(t) and not any(r) for _, _, t, r in self.ranges))


def parse_corrections(data: Dict[str, Any]) -> Corrections:
    """Normalise any of the four dialects into a :class:`Corrections`."""
    default = data.get("default", data)
    translation = _xyz(default.get("translation", _ZERO))
    rotation = _xyz(default.get("rotation", _ZERO))
    entries = (data.get("additional") or data.get("additional_corrections")
               or data.get("corrections") or data.get("fine_corrections") or [])
    ranges = []
    for e in entries:
        start = e.get("start", e.get("start_frame", e.get("start frame", 0)))
        end = e.get("end", e.get("end_frame", e.get("end frame", None)))
        end = int(end) if end is not None else np.iinfo(np.int64).max
        ranges.append((int(start), end,
                       _xyz(e.get("translation", default.get("translation", _ZERO))),
                       _xyz(e.get("rotation", default.get("rotation", _ZERO)))))
    return Corrections(translation, rotation, ranges)


def read_corrections(path: PathLike) -> Corrections:
    """Read a correction JSON in any dialect; a missing file is the identity."""
    p = Path(path)
    if not p.exists():
        return Corrections()
    return parse_corrections(json.loads(p.read_text(encoding="utf-8")))


def corrections_for_frames(corrections: Corrections, n_frames: int
                           ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-frame ``translations (N, 3)`` metres and ``rotations (N, 3)`` radians.

    Frame ``i`` takes the first range that contains it (in file order, as the
    plugin resolves it), otherwise the defaults.
    """
    n = int(n_frames)
    t = np.tile(np.asarray(corrections.translation, float), (n, 1))
    r = np.tile(np.asarray(corrections.rotation, float), (n, 1))
    assigned = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    for start, end, tt, rr in corrections.ranges:
        sel = (idx >= start) & (idx <= end) & ~assigned
        t[sel] = tt
        r[sel] = rr
        assigned |= sel
    return t, r
