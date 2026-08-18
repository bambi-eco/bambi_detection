# -*- coding: utf-8 -*-
"""bambi.survey.transects + bambi.survey.perpendicular - by hand and against the plugin."""
import numpy as np
import pytest

from bambi.survey import perpendicular as pp
from bambi.survey import transects as tr
from tests.plugin_oracle import plugin_module, plugin_processor, requires_plugin


# ---------------------------------------------------------------- transects

def test_cumulative_distances_and_gaps():
    pos = np.array([[0, 0], [3, 4], [np.nan, np.nan], [3, 8], [3, 8]], float)
    cum = tr.cumulative_distances(pos)
    assert cum.tolist() == [0.0, 5.0, 5.0, 9.0, 9.0]
    assert tr.path_length(cum, 3, 0) == 9.0 and tr.path_length(cum, 1, 1) == 0.0
    assert tr.frame_after_distance(cum, 0, 4.0) == 1 and tr.frame_after_distance(cum, 0, 5.0) == 1
    assert tr.frame_after_distance(cum, 1, 3.5) == 3 and tr.frame_after_distance(cum, 0, 100.0) == -1
    assert tr.frame_after_distance(cum, 4, 0.0) == 4
    assert tr.cumulative_distances(np.zeros((0, 2))).shape == (0,)


def test_centerline_skips_nan_and_accepts_either_order():
    pos = np.array([[0, 0, 5], [1, 0, 5], [np.nan, np.nan, 5], [3, 0, 5]], float)
    assert np.array_equal(tr.centerline(pos, 3, 0), [[0, 0], [1, 0], [3, 0]])
    assert np.array_equal(tr.centerline(pos, 1, 1), [[1, 0]])


def test_split_by_distance_and_into():
    pos = np.column_stack([np.arange(11.0), np.zeros(11)])       # 1 m per frame, 10 m total
    cum = tr.cumulative_distances(pos)
    r = tr.split_by_distance(cum, 4.0)
    assert r.tolist() == [[0, 4], [5, 9], [10, 10]]
    assert np.allclose(tr.transect_lengths(cum, r), [4, 4, 0])
    r2 = tr.split_into(cum, 2)
    assert r2.tolist() == [[0, 4], [5, 10]]
    assert tr.split_into(cum, 1).tolist() == [[0, 10]]
    assert tr.split_by_distance(cum, 0).shape == (0, 2)


@requires_plugin
def test_matches_the_plugin_transect_geometry():
    ptr = plugin_module("core.transects")
    rng = np.random.default_rng(2)
    pos = np.cumsum(rng.normal(0, 2, size=(60, 2)), axis=0)
    images = [{"location": [x, y, 30.0]} for x, y in pos]
    images[7] = {}                                                  # a frame without a position
    ours = pos.copy(); ours[7] = np.nan
    cum_ref = ptr.cumulative_distances(ptr.flight_positions(images))
    cum = tr.cumulative_distances(ours)
    assert np.allclose(cum, cum_ref)
    for s, m in ((0, 10.0), (5, 33.3), (50, 500.0), (20, 0.0)):
        ref = ptr.frame_after_distance(cum_ref, s, m)
        assert tr.frame_after_distance(cum, s, m) == (-1 if ref is None else ref)
    assert tr.path_length(cum, 3, 40) == pytest.approx(ptr.path_length(cum_ref, 3, 40))


# ---------------------------------------------------------------- perpendicular

ROUTE = np.array([[0, 0], [10, 0], [10, 10], [20, 10]], float)


def test_nearest_on_polyline_clamps_and_reports_segment():
    foot, dist, seg = pp.nearest_on_polyline([[5, 3], [-2, 0], [12, 4], [25, 12]], ROUTE)
    assert np.allclose(foot, [[5, 0], [0, 0], [10, 4], [20, 10]])
    assert np.allclose(dist, [3, 2, 2, np.hypot(5, 2)])
    assert seg.tolist() == [0, 0, 1, 2]
    assert pp.route_length(ROUTE) == 30.0


def test_perpendicular_uses_the_camera_segment_and_an_unclamped_foot():
    # the detection sits beyond the end of segment 0 but the drone was on segment 0
    foot, dist = pp.perpendicular_to_route([[12, 3]], ROUTE, camera_positions=[[8, 0.5]])
    assert np.allclose(foot, [[12, 0]]) and np.allclose(dist, [3.0])
    # without a camera position the nearest clamped point wins (segment 1)
    foot2, dist2 = pp.perpendicular_to_route([[12, 3]], ROUTE)
    assert np.allclose(foot2, [[10, 3]]) and np.allclose(dist2, [2.0])
    with pytest.raises(ValueError):
        pp.perpendicular_to_route([[0, 0], [1, 1]], ROUTE, camera_positions=[[0, 0]])


def test_flat_footprint_geometry():
    fp = pp.flat_footprint([100, 200, 50], [0, 0, 0], fovy=60.0, aspect_ratio=1.0)
    half = 50 * np.tan(np.deg2rad(30))
    assert np.allclose(np.sort(fp[:, 0]), [100 - half] * 2 + [100 + half] * 2)
    assert np.allclose(np.sort(fp[:, 1]), [200 - half] * 2 + [200 + half] * 2)
    rot = pp.flat_footprint([0, 0, 50], [0, 0, 90], fovy=60.0, aspect_ratio=2.0)
    assert np.allclose(np.abs(rot[:, 0]).max(), half, atol=1e-9)             # width and height swapped by the yaw


@requires_plugin
def test_matches_the_plugin_perpendicular_helpers():
    proc = plugin_processor()
    rng = np.random.default_rng(4)
    route = np.cumsum(rng.normal(0, 5, size=(40, 2)), axis=0)
    route[10] = route[9]                                                # a zero-length segment
    pts = route[rng.integers(0, 40, 50)] + rng.normal(0, 6, size=(50, 2))
    cams = route[rng.integers(0, 39, 50)] + rng.normal(0, 0.5, size=(50, 2))
    foot, dist, _ = pp.nearest_on_polyline(pts, route)
    for i, (x, y) in enumerate(pts):
        fx, fy, d = proc._nearest_on_linestring(route.tolist(), x, y)
        assert np.allclose(foot[i], [fx, fy]) and dist[i] == pytest.approx(d)
    foot, dist = pp.perpendicular_to_route(pts, route, cams)
    for i, ((x, y), (cx, cy)) in enumerate(zip(pts, cams)):
        fx, fy, d = proc._nearest_on_fov_linestring(route.tolist(), [], x, y, cx, cy)
        assert np.allclose(foot[i], [fx, fy]) and dist[i] == pytest.approx(d)
    meta = {"location": [12.5, -3.0, 41.0], "rotation": [0.0, 0.0, 123.4], "fovy": [47.0]}
    ref = np.array(proc._compute_frame_fov_polygon(meta, 500.0, 200.0, aspect_ratio=1.5))
    assert np.allclose(pp.flat_footprint([512.5, 197.0, 41.0], [0, 0, 123.4], 47.0, 1.5), ref)
