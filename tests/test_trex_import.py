# -*- coding: utf-8 -*-
"""TRex import: npz -> raw boxes -> undistorted boxes -> ground.

Unit tests on synthetic tracklet files; the private shark flight (the QGIS
plugin's own TRex import run) is the parity oracle in
``test_parity_trex_plugin.py``.
"""
import numpy as np
import pytest

from bambi.io.trex import read_trex_file, read_trex_tracklets

cv2 = pytest.importorskip("cv2")

from bambi.geo.calibration import fovy_after_undistortion, new_camera_matrix, undistort_boxes, undistort_points  # noqa: E402

MTX = [[3604.8, 0.0, 2560.0], [0.0, 3593.8, 1350.0], [0.0, 0.0, 1.0]]
DIST = [-0.00016, 0.0165, 0.00043, -0.00051, 0.0]
RAW = (5120, 2700)
SQ = (2700, 2700)


def _npz(path, tid, frames, pts, conf=None, cls=None, video_size=(5120.0, 2700.0), with_id=True):
    """pts: (n, K, 2) key-points, NaN where absent."""
    pts = np.asarray(pts, float)
    data = {"frame": np.asarray(frames, np.float32),
            "detection_p": np.ones(len(frames), np.float32) if conf is None else np.asarray(conf, np.float32),
            "detection_class": np.zeros(len(frames), np.float32) if cls is None else np.asarray(cls, np.float32),
            "video_size": np.asarray(video_size, float)}
    for k in range(pts.shape[1]):
        data[f"poseX{k}"] = pts[:, k, 0].astype(np.float32)
        data[f"poseY{k}"] = pts[:, k, 1].astype(np.float32)
    if with_id:
        data["id"] = np.array([tid], np.uint64)
    np.savez(path, **data)
    return path


def test_read_one_file_boxes_are_the_keypoint_extent(tmp_path):
    pts = np.array([[[10, 20], [30, 25], [20, 40]],
                    [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],       # unseen -> dropped
                    [[100, 100], [np.nan, 5], [110, 90]]], float)               # partial NaN key-point
    p = _npz(tmp_path / "x_id3.npz", 3, [0, 1, 5], pts, conf=[0.9, np.nan, 0.5], cls=[1, 0, 2])
    frames, tid, boxes, conf, cls, vs = read_trex_file(p)
    assert tid == 3 and vs == (5120, 2700)
    assert frames.tolist() == [0, 5]
    assert np.allclose(boxes, [[10, 20, 30, 40], [100, 90, 110, 100]])
    assert np.allclose(conf, [0.9, 0.5]) and cls.tolist() == [1, 2]


def test_track_id_from_file_name_when_missing(tmp_path):
    p = _npz(tmp_path / "video_id17.npz", 0, [0], np.zeros((1, 2, 2)), with_id=False)
    assert read_trex_file(p)[1] == 17
    p = _npz(tmp_path / "noid.npz", 0, [0], np.zeros((1, 2, 2)), with_id=False)
    assert read_trex_file(p, fallback_id=4)[1] == 4


def test_folder_merges_and_sorts(tmp_path):
    _npz(tmp_path / "a_id1.npz", 1, [2, 0], np.ones((2, 2, 2)))
    _npz(tmp_path / "b_id0.npz", 0, [1, 0], np.ones((2, 2, 2)) * 2)
    t = read_trex_tracklets(tmp_path)
    assert t.frames.tolist() == [0, 0, 1, 2] and t.track_ids.tolist() == [0, 1, 0, 1]
    assert t.video_size == (5120, 2700) and len(t.files) == 2
    with pytest.raises(FileNotFoundError):
        read_trex_tracklets(tmp_path / "empty")


def test_new_camera_matrix_is_the_extractor_recipe():
    ncm = new_camera_matrix(MTX, DIST, RAW, SQ)
    assert ncm[0, 0] == ncm[1, 1]                          # forced same fov
    assert ncm[0, 2] == pytest.approx(SQ[0] / 2, abs=0.6) and ncm[1, 2] == pytest.approx(SQ[1] / 2, abs=0.6)
    plain = new_camera_matrix(MTX, DIST, RAW, SQ, force_same_fov=False)
    assert plain[0, 0] != plain[1, 1]
    # fovy_after_undistortion is the same matrix turned into an angle
    _, fovy, _, _, _ = cv2.calibrationMatrixValues(ncm, SQ, 1, 1)
    assert fovy_after_undistortion(MTX, DIST, RAW, SQ) == pytest.approx(fovy)


def test_undistort_points_maps_the_principal_point_to_the_new_centre():
    ncm = new_camera_matrix(MTX, DIST, RAW, SQ)
    u = undistort_points([[2560.0, 1350.0]], MTX, DIST, RAW, SQ)
    assert np.allclose(u[0], [ncm[0, 2], ncm[1, 2]], atol=1e-3)
    assert undistort_points(np.zeros((0, 2)), MTX, DIST, RAW, SQ).shape == (0, 2)


def test_undistort_boxes_is_the_extent_of_the_undistorted_corners():
    b = np.array([[1000, 500, 1400, 900], [4000, 2000, 4500, 2600]], float)
    got = undistort_boxes(b, MTX, DIST, RAW, SQ)
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in b]).reshape(-1, 2)
    u = undistort_points(corners, MTX, DIST, RAW, SQ).reshape(-1, 4, 2)
    want = np.column_stack([u[:, :, 0].min(1), u[:, :, 1].min(1), u[:, :, 0].max(1), u[:, :, 1].max(1)])
    assert np.allclose(got, want)
    # a raw box left of the square crop lands at negative x - it is outside the extracted frame
    assert undistort_boxes([[0, 1300, 50, 1400]], MTX, DIST, RAW, SQ)[0, 0] < 0
