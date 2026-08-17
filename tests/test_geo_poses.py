# -*- coding: utf-8 -*-
"""bambi.geo.poses - geographic <-> DEM-local pose arrays.

Pins the conventions the whole pipeline rests on. The parity fixture at the
bottom compares against poses the QGIS plugin actually wrote, so the engine's
maths is held to the oracle, not to itself.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")

from bambi.geo.poses import (  # noqa: E402
    Origin, Poses, geographic_to_local, gimbal_to_rotation, grid_convergence,
    local_to_geographic, make_origin, poses_from_geographic, poses_to_geographic,
    rotation_to_gimbal)

# A UTM 33N anchor (Upper Austria) - central meridian 15E, so grid convergence
# is small but nonzero here.
ORIGIN = make_origin(48.0, 14.0, 500.0, 32633)


# ---------------------------------------------------------------- origin

def test_origin_caches_its_projection():
    assert ORIGIN.epsg == 32633
    # 48N 14E in UTM 33N, well-known to the metre
    assert ORIGIN.easting == pytest.approx(425_405, abs=2)
    assert ORIGIN.northing == pytest.approx(5_316_784, abs=2)
    assert ORIGIN.as_dict() == {"latitude": 48.0, "longitude": 14.0, "altitude": 500.0}


# ---------------------------------------------------------------- positions

def test_origin_maps_to_zero():
    p = geographic_to_local([48.0, 14.0, 500.0], ORIGIN)
    assert p.shape == (1, 3)
    assert np.allclose(p, 0.0, atol=1e-6)


def test_north_and_up_are_positive():
    p = geographic_to_local([[48.001, 14.0, 510.0]], ORIGIN)[0]
    assert p[1] > 100 and p[1] < 115         # ~111 m per 0.001 deg lat
    assert abs(p[0]) < 3                     # a shade of easting from convergence
    assert p[2] == pytest.approx(10.0)


def test_round_trip_is_exact():
    lla = np.array([[48.0, 14.0, 500.0], [48.01, 14.02, 620.5], [47.99, 13.97, 480.0]])
    back = local_to_geographic(geographic_to_local(lla, ORIGIN), ORIGIN)
    assert np.allclose(back, lla, atol=1e-9)


def test_accepts_single_pose_and_batches_alike():
    single = geographic_to_local([48.001, 14.001, 500.0], ORIGIN)
    batch = geographic_to_local([[48.001, 14.001, 500.0]] * 3, ORIGIN)
    assert single.shape == (1, 3) and batch.shape == (3, 3)
    assert np.allclose(batch, single)


@pytest.mark.parametrize("bad", [[48.0, 14.0], [[48.0, 14.0, 500.0, 1.0]]])
def test_rejects_wrong_width(bad):
    with pytest.raises(ValueError):
        geographic_to_local(bad, ORIGIN)


# ---------------------------------------------------------------- rotations

def test_nadir_is_tilt_zero():
    assert gimbal_to_rotation([-90.0, 0.0, 51.6]).tolist() == [[0.0, 0.0, 51.6]]


def test_horizon_is_tilt_ninety():
    assert gimbal_to_rotation([0.0, 0.0, 120.0]).tolist() == [[90.0, 0.0, 120.0]]


def test_heading_wraps_into_0_360():
    r = gimbal_to_rotation([[-90.0, 0.0, -10.0], [-90.0, 0.0, 370.0]])
    assert np.allclose(r[:, 2], [350.0, 10.0])


def test_heading_offset_is_added_per_pose():
    r = gimbal_to_rotation([[-90, 0, 100.0], [-90, 0, 200.0]], heading_offset=[1.5, -1.5])
    assert np.allclose(r[:, 2], [101.5, 198.5])


def test_gimbal_round_trip():
    pry = np.array([[-90.0, 0.0, 51.6], [-45.0, 0.0, 359.0], [0.0, 0.0, 0.0], [-13.6, 0.0, 180.0]])
    back = rotation_to_gimbal(gimbal_to_rotation(pry))
    assert np.allclose(back, pry)


def test_rotation_matches_the_extractor_formula():
    """The extractors write ``(gimbal_pitch + 90) % 360``; nothing else."""
    pitch = np.array([-90.0, -80.0, -13.6, 0.0])
    r = gimbal_to_rotation(np.column_stack([pitch, np.zeros(4), np.zeros(4)]))
    assert np.allclose(r[:, 0], (pitch + 90) % 360)


# ---------------------------------------------------------------- convergence

def test_grid_convergence_is_zero_on_the_central_meridian():
    assert abs(grid_convergence([[48.0, 15.0, 0.0]], 32633)[0]) < 1e-3


def test_grid_convergence_magnitude():
    # |convergence| ~ (lon - lon0) * sin(lat): 1 deg * sin(48) = 0.743 deg
    east = grid_convergence([[48.0, 16.0, 0.0]], 32633)[0]
    west = grid_convergence([[48.0, 14.0, 0.0]], 32633)[0]
    assert abs(east) == pytest.approx(0.743, abs=0.02)
    assert abs(west) == pytest.approx(0.743, abs=0.02)
    assert east == pytest.approx(-west)


def test_grid_convergence_sign_by_pointing():
    """The only argument that settles a sign: point true north, check the grid heading.

    A camera at (48N, 16E) facing TRUE north looks at a target 100 m due north.
    Expressed in the DEM-local (grid) frame, the bearing from camera to target
    is what the heading must become - so ``0 + convergence`` must equal it.
    (pyproj's ``meridian_convergence`` carries the opposite sign; do not swap.)
    """
    from pyproj import Geod
    origin = make_origin(48.0, 16.0, 0.0, 32633)
    tlon, tlat, _ = Geod(ellps="WGS84").fwd(16.0, 48.0, 0.0, 100.0)
    target = geographic_to_local([[tlat, tlon, 0.0]], origin)[0]
    grid_heading = np.degrees(np.arctan2(target[0], target[1])) % 360
    conv = grid_convergence([[48.0, 16.0, 0.0]], 32633)[0]
    assert (conv % 360) == pytest.approx(grid_heading, abs=1e-3)
    assert conv < 0                                    # east of 15E: negative


def test_grid_convergence_is_off_by_default_matching_the_extractors():
    lla = np.array([[48.0, 16.0, 500.0]])
    pry = np.array([[-90.0, 0.0, 100.0]])
    plain = poses_from_geographic(lla, pry, ORIGIN)
    corr = poses_from_geographic(lla, pry, ORIGIN, apply_grid_convergence=True)
    assert plain.rotations[0, 2] == pytest.approx(100.0)
    assert corr.rotations[0, 2] == pytest.approx(100.0 - 0.743, abs=0.02)


# ---------------------------------------------------------------- Poses

def test_poses_round_trip():
    lla = np.array([[48.001, 14.002, 550.0], [48.003, 14.001, 560.0]])
    pry = np.array([[-90.0, 0.0, 30.0], [-60.0, 0.0, 300.0]])
    poses = poses_from_geographic(lla, pry, ORIGIN)
    assert isinstance(poses, Poses) and len(poses) == 2
    lla2, pry2 = poses_to_geographic(poses)
    assert np.allclose(lla2, lla, atol=1e-9)
    assert np.allclose(pry2, pry)


def test_poses_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        poses_from_geographic(np.zeros((3, 3)), np.zeros((2, 3)), ORIGIN)


# ---------------------------------------------------------------- oracle

LION = Path(r"C:\Users\P41743\Desktop\calib\Kilian\data\qgis_fixed\poses_t.json")


@pytest.mark.skipif(not LION.exists(), reason="plugin-written poses not on this machine")
def test_reproduces_plugin_positions_bit_exactly():
    """The QGIS plugin wrote these; the engine must land on the same metres."""
    data = json.loads(LION.read_text(encoding="utf-8"))
    o = data["origin"]
    origin = make_origin(o["latitude"], o["longitude"], o["altitude"], 32737)
    images = data["images"]
    lla = np.array([[im["lat"], im["lng"], 0.0] for im in images])
    want = np.array([im["location"][:2] for im in images])
    got = geographic_to_local(lla, origin)[:, :2]
    assert np.abs(got - want).max() == 0.0
