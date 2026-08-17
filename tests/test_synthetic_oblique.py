# -*- coding: utf-8 -*-
"""The oblique-view proof on the synthetic scene.

Markers sit at known world coordinates on rolling terrain. For cameras at
0/45/80 degrees of tilt across headings: project a marker into the frame,
cast that pixel back onto the mesh, and demand the same point back. Then do
the same with the *old* pose rotation (``quaternion_from_eulers(e, 'zyx')``,
heading about the optical axis) and show it only survives at nadir.
"""
import numpy as np
import pytest

pytest.importorskip("pyrr")
pytest.importorskip("alfspy")
pytest.importorskip("trimesh")

from bambi.geo import camera as cam  # noqa: E402
from bambi.geo import georef  # noqa: E402
from bambi.geo.dem import sample_heightfield  # noqa: E402
from bambi.testing.synthetic import forward_vector, look_at_poses, make_scene, visible_markers  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module", autouse=True)
def convention():
    cam.ray_convention(force_probe=True)


def test_markers_lie_on_the_terrain(scene):
    z = sample_heightfield(scene.elevation, scene.cell_size, scene.markers[:, :2], base_elevation=0.0)
    assert np.allclose(z, scene.markers[:, 2])
    hits = georef.pixels_to_world([[1, 1]], cam.camera_from_pose(scene.markers[0] + [0, 0, 50], [0, 0, 0], 10),
                                  2, 2, scene.mesh)
    assert np.all(np.isfinite(hits))


def test_every_camera_looks_at_the_aim_point(scene):
    w, h = scene.frame_size
    for i, camera in enumerate(cam.cameras_from_poses(scene.poses, scene.fovy)):
        px = cam.world_to_pixel(scene.aim, camera, w, h)[0]
        assert np.allclose(px, [w / 2, h / 2], atol=1e-6), scene.poses.rotations[i]
        # and the world ray through the centre goes back to the aim point on the mesh
        hit = georef.pixels_to_world([w / 2, h / 2], camera, w, h, scene.mesh)[0]
        assert np.linalg.norm(hit - scene.aim) < 0.05, (scene.poses.rotations[i], hit, scene.aim)


def test_project_then_unproject_recovers_every_visible_marker_at_every_tilt(scene):
    """The claim the whole family rests on: exact at 0, 45 and 80 degrees."""
    w, h = scene.frame_size
    mesh = scene.mesh
    worst, hidden = {}, 0
    for i, camera in enumerate(cam.cameras_from_poses(scene.poses, scene.fovy)):
        visible, px = visible_markers(scene, i, mesh)
        assert visible.sum() >= 6, ("too few markers in view", scene.poses.rotations[i])
        hidden += int(np.all(np.isfinite(px), axis=1).sum() - visible.sum())
        back = georef.pixels_to_world(px[visible], camera, w, h, mesh)
        err = np.linalg.norm(back - scene.markers[visible], axis=1)
        tilt = scene.poses.rotations[i, 0]
        worst[tilt] = max(worst.get(tilt, 0.0), float(err.max()))
    for tilt, e in worst.items():
        assert e < 1e-3, (tilt, e)               # millimetres: float32 mesh vertices
    # the scene is honest about occlusion: at 80 degrees some markers hide behind hills
    assert hidden > 0


def test_occluded_markers_are_hit_short_of_the_marker(scene):
    """What visibility filtering removes is real occlusion, not projection error."""
    w, h = scene.frame_size
    mesh = scene.mesh
    for i, camera in enumerate(cam.cameras_from_poses(scene.poses, scene.fovy)):
        visible, px = visible_markers(scene, i, mesh)
        inside = np.all(np.isfinite(px), axis=1) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        occluded = inside & ~visible
        if not occluded.any():
            continue
        back = georef.pixels_to_world(px[occluded], camera, w, h, mesh)
        pos = scene.poses.positions[i]
        assert np.all(np.linalg.norm(back - pos, axis=1) < np.linalg.norm(scene.markers[occluded] - pos, axis=1))


def test_the_old_zyx_rotation_only_works_at_nadir(scene):
    """What the lion flight hit: heading about the optical axis. Exact at tilt 0,
    tens of metres off at 45 and 80 degrees."""
    from pyrr import Vector3
    from alfspy.core.rendering import Camera
    from alfspy.core.util.pyrrs import quaternion_from_eulers

    w, h = scene.frame_size
    mesh = scene.mesh
    legacy = cam.ray_convention() == "legacy"
    miss_by_tilt = {}
    for pos, rot in zip(scene.poses.positions, scene.poses.rotations):
        q = quaternion_from_eulers([np.deg2rad(v) for v in rot], "zyx")
        if legacy:
            q = q.conjugate
        old = Camera(fovy=scene.fovy, aspect_ratio=1.0, position=Vector3(pos), rotation=q)
        hit = georef.pixels_to_world([w / 2, h / 2], old, w, h, mesh)[0]
        miss = np.linalg.norm(hit - scene.aim) if np.all(np.isfinite(hit)) else np.inf
        miss_by_tilt.setdefault(rot[0], []).append(miss)
    assert max(miss_by_tilt[0.0]) < 0.05                    # nadir: indistinguishable
    assert min(miss_by_tilt[45.0]) > 20.0                   # oblique: wildly off (or off the mesh)
    assert min(miss_by_tilt[80.0]) > 20.0


def test_look_at_poses_geometry():
    poses = look_at_poses([100, 100, 10], [0, 45], [0, 90], height_above_aim=60)
    assert np.allclose(poses.positions[0], [100, 100, 70])
    d = poses.positions[1] - [100, 100, 10]
    assert d[2] == pytest.approx(60) and d[0] == pytest.approx(-60) and abs(d[1]) < 1e-9
    assert np.allclose(forward_vector(45, 90), [np.sqrt(0.5), 0, -np.sqrt(0.5)])
    with pytest.raises(ValueError):
        look_at_poses([0, 0, 0], [90], [0], 10)
