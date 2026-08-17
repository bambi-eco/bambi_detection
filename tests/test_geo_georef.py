# -*- coding: utf-8 -*-
"""bambi.geo.georef - pixels onto a mesh, aligned with their inputs, misses as NaN.

The ground truth here is analytic: a flat plane at z=0 and a pinhole camera
whose footprint on that plane can be written down by hand.
"""
import numpy as np
import pytest

pytest.importorskip("pyrr")
pytest.importorskip("alfspy")
trimesh = pytest.importorskip("trimesh")

from bambi.geo import camera as cam  # noqa: E402
from bambi.geo import georef  # noqa: E402

W = H = 512
FOVY = 50.0
ALT = 100.0


@pytest.fixture(scope="module")
def plane():
    # A large flat ground plane at z = 0, two triangles.
    return trimesh.Trimesh(vertices=[[-5000, -5000, 0], [5000, -5000, 0], [5000, 5000, 0], [-5000, 5000, 0]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)


@pytest.fixture(scope="module", autouse=True)
def real_convention():
    cam.ray_convention(force_probe=True)


def nadir():
    return cam.camera_from_pose([0, 0, ALT], [0, 0, 0], fovy=FOVY)


def test_centre_pixel_hits_straight_below(plane):
    hit = georef.pixels_to_world([W / 2, H / 2], nadir(), W, H, plane)
    assert hit.shape == (1, 3)
    assert np.allclose(hit[0], [0, 0, 0], atol=1e-6)


def test_frame_corners_span_the_analytic_footprint(plane):
    half = ALT * np.tan(np.deg2rad(FOVY / 2))
    fp = georef.footprint(nadir(), W, H, plane)
    assert fp.shape == (4, 3)
    # Square frame at nadir: the four corners sit at (+-half, +-half) up to one pixel
    # (the border pixel is at w-1, not w).
    tol = half * 2 / W + 1e-6
    assert np.allclose(np.sort(np.abs(fp[:, 0])), [half] * 4, atol=tol)
    assert np.allclose(np.sort(np.abs(fp[:, 1])), [half] * 4, atol=tol)
    assert np.allclose(fp[:, 2], 0.0, atol=1e-6)


def test_footprint_sampling_walks_the_border(plane):
    fp = georef.footprint(nadir(), W, H, plane, samples_per_edge=8)
    assert fp.shape == (32, 3)
    # closed, convex, all on the ground
    assert np.all(np.isfinite(fp)) and np.allclose(fp[:, 2], 0.0, atol=1e-6)


def test_boxes_to_world_shape_and_order(plane):
    boxes = [[100, 100, 200, 300], [0, 0, 511, 511]]
    out = georef.boxes_to_world(boxes, nadir(), W, H, plane)
    assert out.shape == (2, 4, 3)
    single = georef.pixels_to_world([[100, 100], [200, 100], [200, 300], [100, 300]], nadir(), W, H, plane)
    assert np.allclose(out[0], single)


def test_misses_are_nan_and_stay_aligned(plane):
    """Horizon-tilted camera: the top half of the frame looks at the sky."""
    camera = cam.camera_from_pose([0, 0, ALT], [90.0, 0, 0], fovy=FOVY)
    px = [[W / 2, 10], [W / 2, H - 10], [W / 2, 20]]      # sky, ground, sky
    out = georef.pixels_to_world(px, camera, W, H, plane)
    assert out.shape == (3, 3)
    assert np.all(np.isnan(out[0])) and np.all(np.isnan(out[2]))
    assert np.all(np.isfinite(out[1])) and out[1, 1] > 0     # ground ahead, north of the camera


def test_round_trip_pixel_world_pixel(plane):
    """world_to_pixel is the inverse of pixels_to_world, oblique included."""
    for rot in ([0, 0, 33.0], [45.0, 0, 120.0], [76.4, 0, 50.0]):
        camera = cam.camera_from_pose([12.0, -7.0, ALT], rot, fovy=FOVY)
        # rows in the lower half only: at 76.4 deg tilt the top of the frame is above the horizon
        px = np.array([[10, 300], [W / 2, H / 2], [400, 500], [37.5, 260.25]], float)
        world = georef.pixels_to_world(px, camera, W, H, plane)
        assert np.all(np.isfinite(world)), rot
        back = cam.world_to_pixel(world, camera, W, H)
        assert np.allclose(back, px, atol=1e-3), (rot, back, px)


def test_empty_and_bad_shapes(plane):
    assert georef.pixels_to_world(np.zeros((0, 2)), nadir(), W, H, plane).shape == (0, 3)
    with pytest.raises(ValueError):
        georef.pixels_to_world([[1, 2, 3]], nadir(), W, H, plane)
    with pytest.raises(ValueError):
        georef.boxes_to_world([[1, 2, 3]], nadir(), W, H, plane)


def test_legacy_variant_truncates_and_drops(plane):
    camera = cam.camera_from_pose([0, 0, ALT], [90.0, 0, 0], fovy=FOVY)
    out = georef.pixels_to_world_legacy([W / 2, W / 2 + 0.9], [10, H - 10], W, H, plane, camera)
    assert out.shape == (1, 3)      # the sky pixel is dropped, not NaN
    exact = georef.pixels_to_world([W / 2, H - 10], camera, W, H, plane)   # 256.9 truncated to 256
    assert np.allclose(out[0], exact[0])
