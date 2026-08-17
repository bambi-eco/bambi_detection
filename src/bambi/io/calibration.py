# -*- coding: utf-8 -*-
"""The file edge for camera calibrations: JSON in, media resolution probing.

The compute lives in :mod:`bambi.geo.calibration` and takes arrays; this module
reads the calibration JSON the pipeline uses (``{"mtx": [[...]], "dist": [...]}``)
and asks a video or photo for its size, then hands both to the check.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from bambi.geo.calibration import CalibrationCheck, check_resolution

PathLike = Union[str, Path]

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v")


class CalibrationMismatchError(RuntimeError):
    """Raised when a calibration clearly belongs to a different camera."""


def load_calibration(path: PathLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read ``(mtx, dist)`` arrays from a calibration JSON.

    :param path: JSON with ``mtx`` (3x3) and ``dist`` (flat or 1xN)
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return calibration_arrays(data)


def calibration_arrays(calibration: Dict[str, Any]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(mtx, dist)`` arrays from an already-loaded calibration dict."""
    mtx = np.asarray(calibration["mtx"], dtype=np.float64)
    dist = np.asarray(calibration.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
    return mtx, dist


def media_resolution(paths: Sequence[PathLike]) -> Optional[Tuple[int, int]]:
    """``(width, height)`` of the first readable video or image in *paths*."""
    import cv2

    for path in paths or ():
        path = str(path)
        if not path or not os.path.isfile(path):
            continue
        if os.path.splitext(path)[1].lower() in VIDEO_SUFFIXES:
            cap = cv2.VideoCapture(path)
            try:
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if w > 0 and h > 0:
                        return w, h
            finally:
                cap.release()
        else:
            img = cv2.imread(path)
            if img is not None:
                return img.shape[1], img.shape[0]
    return None


def describe(check: CalibrationCheck, width: int, height: int, mtx: NDArray,
             label: str = "calibration") -> str:
    """Human-readable account of a mismatch, for logs and error messages."""
    text = (
        f"The {label} looks like it was made for a "
        f"{check.implied_width:.0f}x{check.implied_height:.0f} image, but the media is "
        f"{width}x{height} ({check.deviation * 100:.0f}% off).\n"
        f"Its principal point is ({mtx[0, 2]:.1f}, {mtx[1, 2]:.1f}); for this media it "
        f"should be near ({width / 2:.1f}, {height / 2:.1f})."
    )
    if check.severity == "error":
        text += (
            "\n\nApplying it would re-centre every extracted frame on the wrong pixel and "
            f"scale the field of view by roughly {check.scale:.2f}x, so the frames - and "
            "every detection made on them - would point in the wrong direction.\n\n"
            "Pick the calibration that matches this camera, or supply one made from this "
            "camera's own footage."
        )
    return text


def enforce_calibration(calibration: Dict[str, Any], paths: Sequence[PathLike],
                        label: str = "calibration", log_fn=None,
                        allow_mismatch: bool = False) -> Optional[CalibrationCheck]:
    """Run the resolution check for a calibration against media and act on it.

    :param calibration: loaded calibration dict
    :param paths: video/photo paths the calibration will be applied to
    :param label: what to call the calibration in messages
    :param log_fn: optional logging callback
    :param allow_mismatch: downgrade a fatal mismatch to a warning
    :return: the check, or ``None`` if the media could not be read
    :raises CalibrationMismatchError: on a gross mismatch unless *allow_mismatch*
    """
    res = media_resolution(paths)
    if res is None:
        return None
    mtx, _ = calibration_arrays(calibration)
    check = check_resolution(mtx, res[0], res[1])
    if check.ok:
        return check
    message = describe(check, res[0], res[1], mtx, label)
    if check.severity == "error" and not allow_mismatch:
        raise CalibrationMismatchError(message)
    if log_fn:
        log_fn("Warning: " + message)
    return check
