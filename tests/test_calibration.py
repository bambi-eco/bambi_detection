# -*- coding: utf-8 -*-
"""bambi.geo.calibration / bambi.io.calibration - the wrong-calibration guard.

The fixtures are the real thing: the DJI M3T thermal calibration that was
applied to an M30T 1280x1024 stream on the Ol Pejeta lion flight, and the M30T
calibration that should have been. ``test_reproduces_the_lion_fovy`` pins the
number that started the whole investigation.
"""
import numpy as np
import pytest

from bambi.geo.calibration import (check_resolution, fovy_after_undistortion,
                                   implied_resolution)

# DJI M3T (T;Video) - a 640x512 sensor
M3T_MTX = np.array([[762.1973876953125, 0, 315.58062744140625],
                    [0, 745.9588012695312, 258.6109619140625],
                    [0, 0, 1]])
M3T_DIST = np.array([-0.3735257089138031, 0.21459056437015533,
                     -0.0011555751552805305, 0.0010070821736007929, -0.020840825513005257])
# DJI M30T thermal - a 1280x1024 sensor (the lion team's cameraCalib.json)
M30T_MTX = np.array([[1416.32, 0, 636.032], [0, 1416.32, 511.872], [0, 0, 1]])
M30T_DIST = np.array([-0.2177, -0.0249, -0.0019, -0.0015, 0.0009])


# ---------------------------------------------------------------- compute

def test_implied_resolution_is_twice_the_principal_point():
    assert implied_resolution(M30T_MTX) == pytest.approx((1272.064, 1023.744))
    assert implied_resolution(M3T_MTX) == pytest.approx((631.16, 517.22), abs=0.01)


def test_implied_resolution_rejects_bad_matrices():
    with pytest.raises(ValueError):
        implied_resolution(np.eye(2))
    with pytest.raises(ValueError):
        implied_resolution(np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]))


def test_matching_calibration_is_ok():
    c = check_resolution(M30T_MTX, 1280, 1024)
    assert c.ok and c.severity == "ok"
    assert c.deviation < 0.01 and c.scale == pytest.approx(1.0, abs=0.01)


def test_the_lion_mismatch_is_fatal():
    """M3T thermal calibration on an M30T 1280x1024 stream."""
    c = check_resolution(M3T_MTX, 1280, 1024)
    assert c.severity == "error"
    assert c.deviation == pytest.approx(0.507, abs=0.005)
    assert c.implied_width == pytest.approx(631.2, abs=0.1)
    assert c.scale == pytest.approx(0.49, abs=0.01)     # focal length ~halved


def test_moderate_offset_warns():
    mtx = np.array([[1000, 0, 544.0], [0, 1000, 435.0], [0, 0, 1]])   # ~15% off
    assert check_resolution(mtx, 1280, 1024).severity == "warn"


def test_media_size_must_be_positive():
    with pytest.raises(ValueError):
        check_resolution(M30T_MTX, 0, 1024)


def test_reproduces_the_lion_fovy():
    """59.23458155149718 - the exact value in the mis-extracted poses file.

    It reproduces from exactly one combination: the M3T calibration on a
    1280x1024 source, squared to 1024x1024, alpha 0.5, principal point centred,
    fx=fy forced. That is how the wrong preset was identified.
    """
    pytest.importorskip("cv2")
    got = fovy_after_undistortion(M3T_MTX, M3T_DIST, (1280, 1024), (1024, 1024))
    assert got == pytest.approx(59.23458155149718, abs=1e-9)


def test_correct_calibration_gives_the_corrected_fovy():
    pytest.importorskip("cv2")
    got = fovy_after_undistortion(M30T_MTX, M30T_DIST, (1280, 1024), (1024, 1024))
    assert got == pytest.approx(46.375400893756826, abs=1e-9)   # the re-extracted poses carry this


# ---------------------------------------------------------------- edge

def test_enforce_raises_on_gross_mismatch(monkeypatch):
    from bambi.io import calibration as edge
    monkeypatch.setattr(edge, "media_resolution", lambda paths: (1280, 1024))
    with pytest.raises(edge.CalibrationMismatchError) as exc:
        edge.enforce_calibration({"mtx": M3T_MTX.tolist(), "dist": M3T_DIST.tolist()},
                                 ["x.mp4"], "thermal calibration")
    assert "631x517" in str(exc.value) and "1280x1024" in str(exc.value)


def test_enforce_can_be_overridden(monkeypatch):
    from bambi.io import calibration as edge
    monkeypatch.setattr(edge, "media_resolution", lambda paths: (1280, 1024))
    logged = []
    c = edge.enforce_calibration({"mtx": M3T_MTX.tolist(), "dist": M3T_DIST.tolist()},
                                 ["x.mp4"], log_fn=logged.append, allow_mismatch=True)
    assert c.severity == "error" and logged and logged[0].startswith("Warning")


def test_enforce_is_silent_when_it_fits(monkeypatch):
    from bambi.io import calibration as edge
    monkeypatch.setattr(edge, "media_resolution", lambda paths: (1280, 1024))
    logged = []
    c = edge.enforce_calibration({"mtx": M30T_MTX.tolist(), "dist": M30T_DIST.tolist()},
                                 ["x.mp4"], log_fn=logged.append)
    assert c.ok and logged == []


def test_enforce_skips_unreadable_media(monkeypatch):
    from bambi.io import calibration as edge
    monkeypatch.setattr(edge, "media_resolution", lambda paths: None)
    assert edge.enforce_calibration({"mtx": M3T_MTX.tolist()}, ["missing.mp4"]) is None


def test_load_calibration_round_trip(tmp_path):
    import json
    from bambi.io.calibration import load_calibration
    p = tmp_path / "calib.json"
    p.write_text(json.dumps({"mtx": M30T_MTX.tolist(), "dist": M30T_DIST.tolist()}))
    mtx, dist = load_calibration(p)
    assert np.allclose(mtx, M30T_MTX) and np.allclose(dist, M30T_DIST)


# --- undistort_maps / remap: the "cv2.error: Unknown exception" guard ------

W_MTX = np.array([[2888.17822265625, 0.0, 1929.0179443359375],
                  [0.0, 2819.316162109375, 1070.7803955078125],
                  [0.0, 0.0, 1.0]])
W_DIST = np.array([0.13853585720062256, -0.25508561730384827,
                   0.0002033660130109638, -0.0009057472343556583, 0.0])


def _w_maps():
    from bambi.geo.calibration import new_camera_matrix, undistort_maps
    ncm = new_camera_matrix(W_MTX, W_DIST, (3840, 2160), (2160, 2160))
    return undistort_maps(W_MTX, W_DIST, ncm, (2160, 2160)), ncm


def test_undistort_maps_matches_opencv():
    import cv2

    (mapx, mapy), ncm = _w_maps()
    ex, ey = cv2.initUndistortRectifyMap(W_MTX, W_DIST.reshape(1, -1), None, ncm,
                                         (2160, 2160), cv2.CV_32FC1)
    assert mapx.shape == mapy.shape == (2160, 2160)
    assert mapx.dtype == np.float32
    np.testing.assert_array_equal(mapx, ex)
    np.testing.assert_array_equal(mapy, ey)


def test_remap_matches_opencv():
    import cv2
    from bambi.geo.calibration import remap

    (mapx, mapy), _ = _w_maps()
    img = np.random.default_rng(0).integers(0, 255, (2160, 3840, 3), dtype=np.uint8)
    np.testing.assert_array_equal(remap(img, mapx, mapy, cv2.INTER_LINEAR),
                                  cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR))


def test_undistort_maps_retries_single_threaded(monkeypatch, caplog):
    """A parallel-backend failure is retried with OpenCV threads disabled."""
    import logging

    import cv2
    from bambi.geo import calibration

    real = cv2.initUndistortRectifyMap
    calls = []

    def flaky(*args, **kwargs):
        calls.append(cv2.getNumThreads())
        if len(calls) == 1:
            raise cv2.error("Unknown exception")
        return real(*args, **kwargs)

    set_calls = []
    monkeypatch.setattr(cv2, "initUndistortRectifyMap", flaky)
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: set_calls.append(n))
    with caplog.at_level(logging.WARNING, logger="bambi.geo.calibration"):
        mapx, _ = calibration.undistort_maps(W_MTX, W_DIST, W_MTX, (64, 48))
    assert mapx.shape == (48, 64)
    assert len(calls) == 2
    assert set_calls == [1]
    assert "Unknown exception" in caplog.text
    assert "single-threaded" in caplog.text


def test_undistort_maps_reports_inputs_when_retry_fails(monkeypatch):
    import cv2
    from bambi.geo import calibration

    def broken(*args, **kwargs):
        raise cv2.error("Unknown exception")

    monkeypatch.setattr(cv2, "initUndistortRectifyMap", broken)
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: None)
    with pytest.raises(RuntimeError) as info:
        calibration.undistort_maps(W_MTX, W_DIST, W_MTX, (640, 512))
    msg = str(info.value)
    assert "initUndistortRectifyMap" in msg
    assert "new_size=(640, 512)" in msg
    assert "2888.178" in msg
    assert "Unknown exception" in msg
    assert "OpenCV " + cv2.__version__ in msg
    assert isinstance(info.value.__cause__, cv2.error)
