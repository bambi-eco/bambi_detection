# -*- coding: utf-8 -*-
"""bambi.geo.camera - poses become cameras that point where the gimbal pointed.

Ported from the QGIS plugin's ``tests_qgis/test_camera_pose_conventions.py``
so the engine carries the same proofs: heading about world up after the tilt,
both alfspy ray conventions cast the same world ray, the 1x correction rule.
Runs against whichever alfspy backend is importable (alfs_py or alfs_pytorch).
"""
import numpy as np
import pytest

pytest.importorskip("pyrr")
pytest.importorskip("alfspy")

from bambi.geo import camera as cam  # noqa: E402
from bambi.geo.poses import Poses, make_origin  # noqa: E402

# [tilt off nadir, roll, heading clockwise from north]
POSES = [
    [0.0, 0.0, 51.6],       # nadir, yawed
    [90.0, 0.0, 51.6],      # horizon, north-east
    [45.0, 0.0, 120.0],     # oblique
    [7.3, 0.0, 15.4],       # near nadir
    [60.0, 0.0, -140.0],    # oblique, negative heading
    [76.4, 0.0, 50.0],      # the lion flight's dominant oblique pose
]


def expected_forward(tilt, heading):
    """ENU direction of a gimbal tilted *tilt* off nadir on *heading*."""
    t, h = np.radians(tilt), np.radians(heading)
    return np.array([np.sin(t) * np.sin(h), np.sin(t) * np.cos(h), -np.cos(t)])


def angle_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def camera_forward(camera, convention):
    rot = np.asarray(camera.transform.rotation.matrix33, dtype=float)
    return np.array([0.0, 0.0, -1.0]) @ (rot.T if convention == "legacy" else rot)


@pytest.mark.parametrize("rot", POSES)
@pytest.mark.parametrize("convention", ["legacy", "fixed"])
def test_camera_points_where_the_gimbal_pointed(rot, convention, monkeypatch):
    monkeypatch.setattr(cam, "_RAY_CONVENTION", convention)
    camera = cam.camera_from_pose([0.0, 0.0, 100.0], rot, fovy=50.0)
    assert angle_between(camera_forward(camera, convention), expected_forward(rot[0], rot[2])) < 1e-3


@pytest.mark.parametrize("rot", POSES)
def test_both_conventions_cast_the_same_world_ray(rot, monkeypatch):
    tan_fov = np.tan(np.deg2rad(50.0) / 2.0)
    local = np.array([0.6 * tan_fov, -0.35 * tan_fov, -1.0])   # off-centre: roll errors would show
    rays = {}
    for convention in ("legacy", "fixed"):
        monkeypatch.setattr(cam, "_RAY_CONVENTION", convention)
        rot33 = np.asarray(cam.camera_from_pose([0, 0, 100], rot, 50.0).transform.rotation.matrix33, float)
        ray = local @ (rot33.T if convention == "legacy" else rot33)
        rays[convention] = ray / np.linalg.norm(ray)
    assert np.allclose(rays["legacy"], rays["fixed"], atol=1e-6)


def test_heading_steers_the_tilt(monkeypatch):
    """The regression: two headings at the same tilt must differ."""
    monkeypatch.setattr(cam, "_RAY_CONVENTION", "fixed")
    north = camera_forward(cam.camera_from_pose([0, 0, 100], [45.0, 0.0, 0.0], 50.0), "fixed")
    east = camera_forward(cam.camera_from_pose([0, 0, 100], [45.0, 0.0, 90.0], 50.0), "fixed")
    assert angle_between(north, east) == pytest.approx(60.0, abs=1e-2)
    assert north[0] == pytest.approx(0.0, abs=1e-5)
    assert east[1] == pytest.approx(0.0, abs=1e-5)


def test_correction_is_applied_once_and_in_radians(monkeypatch):
    """A 1 degree heading correction must move the camera by 1 degree, not 2."""
    monkeypatch.setattr(cam, "_RAY_CONVENTION", "fixed")
    plain = camera_forward(cam.camera_from_pose([0, 0, 100], [90.0, 0, 0], 50.0), "fixed")
    nudged = camera_forward(cam.camera_from_pose([0, 0, 100], [90.0, 0, 0], 50.0,
                                                 rotation_correction=[0, 0, np.deg2rad(1.0)]), "fixed")
    assert angle_between(plain, nudged) == pytest.approx(1.0, abs=1e-3)


def test_translation_correction_moves_the_camera(monkeypatch):
    monkeypatch.setattr(cam, "_RAY_CONVENTION", "fixed")
    camera = cam.camera_from_pose([10, 20, 30], [0, 0, 0], 50.0, translation_correction=[1, 2, 3])
    assert np.allclose(np.asarray(camera.transform.position, float), [11, 22, 33])


def test_probe_matches_the_installed_alfspy():
    pytest.importorskip("trimesh")
    convention = cam.ray_convention(force_probe=True)
    assert convention in ("legacy", "fixed")
    import inspect
    from alfspy.core.convert.convert import pixel_to_world_coord
    source = inspect.getsource(pixel_to_world_coord)
    assert convention == ("legacy" if "matrix33.T" in source else "fixed")


def test_probe_result_is_cached(monkeypatch):
    monkeypatch.setattr(cam, "_RAY_CONVENTION", "sentinel")
    assert cam.ray_convention() == "sentinel"


def _poses(n=3):
    return Poses(positions=np.array([[0, 0, 100.0]] * n), rotations=np.array(POSES[:n], float),
                 origin=make_origin(46.0, 14.0, 400.0, 32633))


def test_cameras_from_poses_batches_and_subsets(monkeypatch):
    monkeypatch.setattr(cam, "_RAY_CONVENTION", "fixed")
    cams = cam.cameras_from_poses(_poses(), fovy=50.0)
    assert len(cams) == 3
    sub = cam.cameras_from_poses(_poses(), fovy=[50.0, 40.0, 30.0], indices=[2])
    assert len(sub) == 1 and sub[0].fovy == pytest.approx(30.0)
    assert angle_between(camera_forward(sub[0], "fixed"), expected_forward(45.0, 120.0)) < 1e-3


def test_cameras_from_poses_rejects_bad_correction_shapes(monkeypatch):
    monkeypatch.setattr(cam, "_RAY_CONVENTION", "fixed")
    with pytest.raises(ValueError):
        cam.cameras_from_poses(_poses(), 50.0, translation_corrections=np.zeros((2, 3)))


def test_world_to_pixel_hits_the_principal_point_and_beyond(monkeypatch):
    """A nadir camera at 100 m: the ground point straight below is the frame centre,
    a point 100*tan(fovy/2) m north is the top edge; behind the camera -> NaN."""
    monkeypatch.setattr(cam, "_RAY_CONVENTION", cam.ray_convention(force_probe=True))
    camera = cam.camera_from_pose([0, 0, 100.0], [0, 0, 0], fovy=50.0)
    half = 100.0 * np.tan(np.deg2rad(25.0))
    px = cam.world_to_pixel([[0, 0, 0], [0, half, 0], [half, 0, 0], [0, 0, 200.0]], camera, 512, 512)
    assert np.allclose(px[0], [256, 256], atol=1e-6)
    # the two edge points land on the frame border (which border is a convention
    # pinned by the round-trip test in test_geo_georef)
    for p in px[1:3]:
        assert min(abs(p[0]), abs(p[0] - 512), abs(p[1]), abs(p[1] - 512)) < 1e-4
    assert np.all(np.isnan(px[3]))
