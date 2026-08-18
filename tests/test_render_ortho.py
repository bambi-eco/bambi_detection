# -*- coding: utf-8 -*-
"""bambi.render - orthophotos land where the ray caster says, at any tilt.

Synthetic scene, a frame image with a bright marker at a known pixel: the
orthophoto of that frame must show the marker at the ground position
``pixels_to_world`` gives for that pixel - render and ray cast share one
pose convention. Then the integral of several frames, tiling, erosion.
"""
import numpy as np
import pytest

pytest.importorskip("pyrr")
pytest.importorskip("alfspy")
pytest.importorskip("trimesh")
cv2 = pytest.importorskip("cv2")

from bambi.geo import camera as cam  # noqa: E402
from bambi.geo import georef  # noqa: E402
from bambi.io.dem import render_data_from_arrays  # noqa: E402
from bambi.render import masks, ortho  # noqa: E402
from bambi.testing.synthetic import make_scene  # noqa: E402
from bambi.util.render_context import make_render_context  # noqa: E402

FRAME = (256, 256)
FOVY = 50.0


@pytest.fixture(scope="module")
def scene():
    return make_scene(tilts_deg=(0.0, 45.0), headings_deg=(30.0, 200.0), frame_size=FRAME, fovy=FOVY)


@pytest.fixture(scope="module")
def render_data(scene):
    return render_data_from_arrays(scene.vertices, scene.faces)


@pytest.fixture(scope="module")
def ctx():
    c = make_render_context()
    yield c
    try:
        from alfspy.render.render import release_all
        release_all(c)
    except Exception:
        pass


def _marker_frame(px, py, size=6):
    img = np.full((FRAME[1], FRAME[0], 3), 40, dtype=np.uint8)
    img[py - size:py + size, px - size:px + size] = (255, 255, 255)        # white marker
    return img


@pytest.mark.parametrize("pose_index", [0, 1, 2, 3])
def test_orthophoto_puts_the_marker_where_the_ray_lands(scene, render_data, ctx, pose_index):
    mesh_data, texture_data = render_data
    pos, rot = scene.poses.positions[pose_index], scene.poses.rotations[pose_index]
    px, py = 96, 150
    camera = cam.camera_from_pose(pos, rot, FOVY, FRAME[0] / FRAME[1])
    truth = georef.pixels_to_world([px, py], camera, FRAME[0], FRAME[1], scene.mesh)[0]
    fp = georef.footprint(camera, FRAME[0], FRAME[1], scene.mesh, samples_per_edge=4)
    bounds = masks.polygon_bounds(fp)
    assert bounds is not None and np.all(np.isfinite(truth))
    size = ortho.render_size(bounds[2] - bounds[0], bounds[3] - bounds[1], ground_resolution=0.25, max_dim=1024)
    shot = ortho.make_shot(ctx, _marker_frame(px, py), pos, rot, FOVY, FRAME[0] / FRAME[1], lazy=False)
    img = ortho.render_orthographic(ctx, mesh_data, texture_data, shot, bounds, size)
    assert img.shape == (size[1], size[0], 4) and img.dtype == np.uint8
    assert (img[:, :, 3] > 0).mean() > 0.2                                # the footprint is there
    bright = img[:, :, :3].min(axis=2) > 200
    assert bright.any()
    rows, cols = np.nonzero(bright)
    cx = bounds[0] + (cols.mean() + 0.5) * (bounds[2] - bounds[0]) / size[0]
    cy = bounds[3] - (rows.mean() + 0.5) * (bounds[3] - bounds[1]) / size[1]
    assert np.hypot(cx - truth[0], cy - truth[1]) < 1.0, (rot, (cx, cy), truth)   # within a metre on rolling terrain


def test_integral_of_two_shots_covers_the_union_and_averages(scene, render_data, ctx):
    mesh_data, texture_data = render_data
    a, b = 0, 1
    shots = [ortho.make_shot(ctx, np.full((FRAME[1], FRAME[0], 3), v, np.uint8), scene.poses.positions[i],
                             scene.poses.rotations[i], FOVY, 1.0, lazy=False) for i, v in ((a, 60), (b, 200))]
    fps = [georef.footprint(cam.camera_from_pose(scene.poses.positions[i], scene.poses.rotations[i], FOVY), FRAME[0],
                            FRAME[1], scene.mesh) for i in (a, b)]
    allp = np.vstack(fps)
    bounds = masks.polygon_bounds(allp)
    size = ortho.render_size(bounds[2] - bounds[0], bounds[3] - bounds[1], 0.5)
    integral = ortho.render_orthographic(ctx, mesh_data, texture_data, shots, bounds, size, integral=True,
                                         auto_contrast=False)
    single_a = ortho.render_orthographic(ctx, mesh_data, texture_data, shots[0], bounds, size)
    single_b = ortho.render_orthographic(ctx, mesh_data, texture_data, shots[1], bounds, size)
    cov = (integral[:, :, 3] > 0)
    assert cov.sum() >= max((single_a[:, :, 3] > 0).sum(), (single_b[:, :, 3] > 0).sum())
    both = (single_a[:, :, 3] > 0) & (single_b[:, :, 3] > 0)
    only_a = (single_a[:, :, 3] > 0) & ~(single_b[:, :, 3] > 0)
    if both.sum() > 50 and only_a.sum() > 50:
        # where both frames contribute the integral is between the two grey levels
        assert 60 <= np.median(integral[:, :, 0][both]) <= 200
        assert np.median(integral[:, :, 0][only_a]) < np.median(integral[:, :, 0][both]) + 5


def test_tiles_and_tile_camera():
    t = ortho.tiles((1000, 700), 400)
    assert t.tolist() == [[0, 0, 400, 400], [400, 0, 400, 400], [800, 0, 200, 400],
                          [0, 400, 400, 300], [400, 400, 400, 300], [800, 400, 200, 300]]
    b = ortho.tile_bounds((0, 0, 100, 70), (1000, 700), [800, 400, 200, 300])
    assert b == pytest.approx((80, 0, 100, 30))
    g = ortho.ortho_camera((0, 0, 100, 70), 500.0)
    tc = ortho.tile_camera(g, (0, 0, 100, 70), (1000, 700), [800, 400, 200, 300])
    assert tuple(tc.orthogonal_size) == pytest.approx((20, 30))
    assert np.allclose(np.asarray(tc.transform.position, float), [90, 15, 500])


def test_render_size_and_erosion_and_crop():
    assert ortho.render_size(10.0, 4.5, 0.5) == (20, 9)
    assert ortho.render_size(10.0, 4.5, fixed=(2048, 1024)) == (2048, 1024)
    assert ortho.render_size(10.0, 4.5, 0.001, max_dim=100) == (100, 45)
    with pytest.raises(ValueError):
        ortho.render_size(1, 1, 0)
    m = np.zeros((7, 7), bool)
    m[1:6, 1:6] = True
    e = ortho.erode_valid_mask(m, 1)
    assert e.sum() == 9 and e[3, 3] and not e[1, 1]
    assert ortho.erode_valid_mask(m, 0) is m or np.array_equal(ortho.erode_valid_mask(m, 0), m)
    img = np.zeros((10, 20, 4), np.uint8)
    img[2:5, 4:9, 3] = 255
    crop, b = ortho.crop_to_content(img, (0, 0, 20, 10))
    assert crop.shape == (3, 5, 4) and b == pytest.approx((4, 5, 9, 8))


def test_mask_polygon_and_default():
    mask = np.zeros((100, 120), np.uint8)
    cv2.circle(mask, (60, 50), 30, 255, -1)
    poly = masks.mask_polygon(mask, simplify_epsilon=2.0)
    assert poly is not None and poly.shape[1] == 2 and 8 <= len(poly) <= 60
    assert abs(poly[:, 0].mean() - 60) < 3 and abs(poly[:, 1].mean() - 50) < 3
    assert masks.mask_polygon(np.zeros((10, 10), np.uint8)) is None
    d = masks.default_mask_polygon(640, 512)
    assert d.shape == (8, 2) and d[4].tolist() == [640, 512]
    assert masks.polygon_bounds([[0, 0], [1, 1]]) is None
    assert masks.polygon_bounds([[0, 0, 5], [2, 1, 5], [np.nan, np.nan, 5], [1, 3, 5]]) == (0, 0, 2, 3)


def test_polygon_to_world_legacy_matches_the_one_pixel_box_recipe(scene):
    pos, rot = scene.poses.positions[0], scene.poses.rotations[0]
    camera = cam.camera_from_pose(pos, rot, FOVY)
    poly = masks.default_mask_polygon(*FRAME)[[0, 2, 4, 6]] + [[1, 1], [-2, 1], [-2, -2], [1, -2]]   # inside the frame
    got = masks.polygon_to_world(poly, camera, FRAME[0], FRAME[1], scene.mesh, legacy=True)
    for i, (px, py) in enumerate(poly):
        ref = georef.pixels_to_world_legacy([px, px + 1, px + 1, px], [py, py, py + 1, py + 1], FRAME[0], FRAME[1],
                                            scene.mesh, camera).mean(axis=0)
        assert np.allclose(got[i], ref)
    exact = masks.polygon_to_world(poly, camera, FRAME[0], FRAME[1], scene.mesh, legacy=False)
    assert np.all(np.isfinite(exact)) and np.linalg.norm(exact - got, axis=1).max() < 0.5


def test_tiled_integral_equals_the_untiled_one(scene, render_data, ctx):
    mesh_data, texture_data = render_data
    idx = [0, 1]
    shots = [ortho.make_shot(ctx, np.full((FRAME[1], FRAME[0], 3), v, np.uint8), scene.poses.positions[i],
                             scene.poses.rotations[i], FOVY, 1.0, lazy=False) for i, v in zip(idx, (80, 180))]
    fps = [georef.footprint(cam.camera_from_pose(scene.poses.positions[i], scene.poses.rotations[i], FOVY), FRAME[0],
                            FRAME[1], scene.mesh) for i in idx]
    bounds = masks.polygon_bounds(np.vstack(fps))
    size = ortho.render_size(bounds[2] - bounds[0], bounds[3] - bounds[1], 0.5)
    whole = ortho.render_orthographic(ctx, mesh_data, texture_data, shots, bounds, size, integral=True,
                                      auto_contrast=False)
    tiled, tb = ortho.render_integral_tiled(ctx, mesh_data, texture_data, shots, bounds, size, max_tile=64,
                                            auto_contrast=False, shot_footprints=fps)
    assert tb == bounds and tiled.shape == whole.shape
    cov_w, cov_t = whole[:, :, 3] > 0, tiled[:, :, 3] > 0
    assert (cov_w == cov_t).mean() > 0.995                              # same footprint, tile seams aside
    both = cov_w & cov_t
    assert np.abs(tiled[:, :, :3][both].astype(int) - whole[:, :, :3][both].astype(int)).mean() < 3
    cropped, cb = ortho.render_integral_tiled(ctx, mesh_data, texture_data, shots, bounds, size, max_tile=64,
                                              auto_contrast=False, crop=True)
    assert cropped.shape[0] <= tiled.shape[0] and cropped.shape[1] <= tiled.shape[1]
    assert cb[0] >= bounds[0] and cb[2] <= bounds[2]
