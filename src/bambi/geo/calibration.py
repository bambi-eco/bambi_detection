# -*- coding: utf-8 -*-
"""Camera calibration checks, on arrays.

The one that has bitten hardest: applying a calibration made for one sensor
to video from another. OpenCV accepts it silently - ``getOptimalNewCameraMatrix``
and ``initUndistortRectifyMap`` happily map a 640x512 calibration onto a
1280x1024 stream - and the output still *looks* like a frame. But the focal
length is off by the size ratio and, worse, the frame is re-centred on the
calibration's principal point instead of the image centre, so every extracted
frame points tens of degrees away from where the pipeline thinks it does.
On the Ol Pejeta lion flight that put detections a kilometre off.

The tell is cheap: a sane calibration's principal point sits near the centre
of the image it was made on, so ``(2 * cx, 2 * cy)`` recovers that image's
size and can be compared with the media being extracted.

Engine functions here take the 3x3 camera matrix as an array. Reading a
calibration JSON and probing a video's resolution are edge concerns
(:mod:`bambi.io.calibration`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "new_camera_matrix",
    "undistort_points",
    "undistort_boxes",
    "CalibrationCheck",
    "implied_resolution",
    "check_resolution",
    "fovy_after_undistortion",
    "WARN_TOLERANCE",
    "FATAL_TOLERANCE",
]

#: Relative size difference above which a mismatch is merely reported.
WARN_TOLERANCE = 0.10
#: Relative size difference above which extraction should be refused.
FATAL_TOLERANCE = 0.25


@dataclass(frozen=True)
class CalibrationCheck:
    """Result of comparing a calibration with the media it will be applied to.

    :ivar severity: ``"ok"``, ``"warn"`` or ``"error"``
    :ivar deviation: worst relative size difference (0 = perfect)
    :ivar implied_width: image width the calibration was made for
    :ivar implied_height: image height the calibration was made for
    :ivar scale: focal-length scale the mismatch would apply (1 = none)
    """
    severity: str
    deviation: float
    implied_width: float
    implied_height: float
    scale: float

    @property
    def ok(self) -> bool:
        return self.severity == "ok"


def _matrix(mtx: ArrayLike) -> NDArray[np.float64]:
    m = np.asarray(mtx, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"camera matrix must be 3x3, got {m.shape}")
    return m


def implied_resolution(mtx: ArrayLike) -> Tuple[float, float]:
    """Image size a calibration was made for, from its principal point.

    :param mtx: 3x3 intrinsic matrix
    :return: ``(width, height)`` = ``(2*cx, 2*cy)``
    :raises ValueError: if the principal point is not positive
    """
    m = _matrix(mtx)
    cx, cy = m[0, 2], m[1, 2]
    if cx <= 0 or cy <= 0:
        raise ValueError(f"principal point must be positive, got ({cx}, {cy})")
    return 2.0 * cx, 2.0 * cy


def check_resolution(mtx: ArrayLike, width: int, height: int,
                     warn_tolerance: float = WARN_TOLERANCE,
                     fatal_tolerance: float = FATAL_TOLERANCE) -> CalibrationCheck:
    """Compare a calibration against the media it is about to be applied to.

    :param mtx: 3x3 intrinsic matrix
    :param width: media width in pixels
    :param height: media height in pixels
    :param warn_tolerance: relative size difference that yields ``"warn"``
    :param fatal_tolerance: relative size difference that yields ``"error"``
    """
    if not width or not height:
        raise ValueError("media width and height must be positive")
    cal_w, cal_h = implied_resolution(mtx)
    dev_w = abs(cal_w - width) / float(width)
    dev_h = abs(cal_h - height) / float(height)
    deviation = float(max(dev_w, dev_h))
    scale = float(cal_w / width)
    if deviation <= warn_tolerance:
        severity = "ok"
    elif deviation >= fatal_tolerance:
        severity = "error"
    else:
        severity = "warn"
    return CalibrationCheck(severity, deviation, float(cal_w), float(cal_h), scale)


def new_camera_matrix(mtx: ArrayLike, dist: ArrayLike, src_size: Tuple[int, int],
                      new_size: Tuple[int, int], alpha: float = 0.5,
                      force_same_fov: bool = True) -> NDArray[np.float64]:
    """The intrinsics of the *undistorted* frames the extractors produce.

    ``getOptimalNewCameraMatrix`` with the principal point centred, then
    (optionally) fx = fy = max(fx, fy). This is the one recipe the extractors,
    the poses' ``fovy`` and every pixel-space import (TRex) must share; a
    point undistorted with a different matrix lands on the wrong pixel of
    the extracted frame.

    :param mtx: 3x3 intrinsic matrix of the raw camera
    :param dist: distortion coefficients (4, 5 or 8)
    :param src_size: ``(width, height)`` of the raw video/photo
    :param new_size: ``(width, height)`` of the extracted frames
    :param alpha: OpenCV free-scaling parameter (extractors use 0.5)
    :param force_same_fov: set fx = fy = max(fx, fy) like the extractors
    """
    import cv2

    m = _matrix(mtx)
    d = np.asarray(dist, dtype=np.float64).reshape(1, -1)
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(m, d, tuple(int(v) for v in src_size), alpha,
                                               tuple(int(v) for v in new_size),
                                               centerPrincipalPoint=True)
    new_mtx = np.asarray(new_mtx, dtype=np.float64).copy()
    if force_same_fov:
        f = max(new_mtx[0, 0], new_mtx[1, 1])
        new_mtx[0, 0] = new_mtx[1, 1] = f
    return new_mtx


def fovy_after_undistortion(mtx: ArrayLike, dist: ArrayLike, src_size: Tuple[int, int],
                            new_size: Tuple[int, int], alpha: float = 0.5,
                            force_same_fov: bool = True) -> float:
    """Vertical field of view (degrees) of frames the extractors will produce.

    :func:`new_camera_matrix` turned into an angle, so a caller can predict
    the ``fovy`` a poses file will carry, or check one.
    """
    import cv2

    new_mtx = new_camera_matrix(mtx, dist, src_size, new_size, alpha, force_same_fov)
    _, fovy, _, _, _ = cv2.calibrationMatrixValues(new_mtx, tuple(int(v) for v in new_size), 1, 1)
    return float(fovy)


def undistort_points(points: ArrayLike, mtx: ArrayLike, dist: ArrayLike, src_size: Tuple[int, int],
                     new_size: Tuple[int, int], alpha: float = 0.5,
                     force_same_fov: bool = True) -> NDArray[np.float64]:
    """Raw-video pixels -> pixels of the undistorted, extracted frame.

    Maps ``(N, 2)`` ``[x, y]`` measured on the raw video (what an external
    tracker such as TRex sees) into the frame space the poses were built for.
    ``force_same_fov`` must match how the frames were extracted; the
    extractors force it always, the QGIS plugin's TRex import only for
    square frames.

    :return: ``(N, 2)`` float64
    """
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    m = _matrix(mtx)
    d = np.asarray(dist, dtype=np.float64).reshape(1, -1)
    ncm = new_camera_matrix(m, d, src_size, new_size, alpha, force_same_fov)
    out = cv2.undistortPoints(pts, m, d, P=ncm)
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def undistort_boxes(boxes: ArrayLike, mtx: ArrayLike, dist: ArrayLike, src_size: Tuple[int, int],
                    new_size: Tuple[int, int], alpha: float = 0.5,
                    force_same_fov: bool = True) -> NDArray[np.float64]:
    """``(N, 4)`` xyxy raw-video boxes -> axis-aligned boxes around their
    four undistorted corners, in extracted-frame pixels."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0:
        return np.zeros((0, 4))
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1).reshape(-1, 2)
    u = undistort_points(corners, mtx, dist, src_size, new_size, alpha, force_same_fov).reshape(-1, 4, 2)
    return np.column_stack([u[:, :, 0].min(axis=1), u[:, :, 1].min(axis=1),
                            u[:, :, 0].max(axis=1), u[:, :, 1].max(axis=1)])
