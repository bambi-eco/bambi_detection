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
