# -*- coding: utf-8 -*-
"""Parity with the QGIS plugin's "Import TRex Tracklets" on a real flight.

The plugin turned a folder of TRex ``.npz`` files into ``detections.txt``,
``tracks_pixel.csv`` (raw video pixels), ``tracks_pixel_undistorted.csv``
(extracted-frame pixels) and ``tracks.csv`` (ground, on a flat DEM). Each of
those is re-derived here from the same inputs and compared. Private data::

    BAMBI_PARITY_TREX=C:/.../shark      containing tracking/*.npz, qgis/{poses_w.json,
                                        flat_surface_dem.glb, tracks_pixel_w/, tracks_w/, detections_w/}
                                        and the calibration JSON named below
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

ROOT = os.environ.get("BAMBI_PARITY_TREX")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not ROOT, reason="set BAMBI_PARITY_TREX")]
CALIB = "angela_pink_combined.json"
FRAME = (2700, 2700)          # the extracted (square) RGB frames


@pytest.fixture(scope="module")
def data():
    from bambi.io import tracks as io
    from bambi.io.calibration import load_calibration
    from bambi.io.trex import read_trex_tracklets

    root = Path(ROOT)
    trex = read_trex_tracklets(root / "tracking")
    mtx, dist = load_calibration(root / CALIB)
    ref_raw = io.read_pixel_tracks(root / "qgis" / "tracks_pixel_w" / "tracks_pixel.csv")
    ref_und = io.read_pixel_tracks(root / "qgis" / "tracks_pixel_w" / "tracks_pixel_undistorted.csv")
    ref_geo = io.read_tracks_csv(root / "qgis" / "tracks_w" / "tracks.csv")
    ref_det = io.read_detections(root / "qgis" / "detections_w" / "detections.txt")
    return root, trex, (mtx, dist), ref_raw, ref_und, ref_geo, ref_det


def test_npz_boxes_match_the_plugins_raw_pixel_tracks(data):
    _, trex, _, ref_raw, _, _, ref_det = data
    assert len(trex) == len(ref_raw) == len(ref_det)
    assert np.array_equal(trex.frames, ref_raw.frames) and np.array_equal(trex.track_ids, ref_raw.track_ids)
    assert np.abs(trex.boxes - ref_raw.boxes).max() < 1e-5
    assert np.abs(trex.confidences - ref_raw.confidences).max() < 1e-5
    assert np.array_equal(trex.classes, ref_raw.classes)
    assert np.abs(trex.boxes - ref_det.boxes).max() < 0.006      # detections.txt has 2 decimals


def test_undistortion_matches_the_plugin(data):
    from bambi.geo.calibration import undistort_boxes

    _, trex, (mtx, dist), _, ref_und, _, _ = data
    assert trex.video_size == (5120, 2700)
    got = undistort_boxes(trex.boxes, mtx, dist, trex.video_size, FRAME)
    assert np.abs(got - ref_und.boxes).max() < 1e-4


def test_ground_tracks_match_the_plugin(data):
    from bambi.geo.calibration import undistort_boxes
    from bambi.geo.georef import boxes_to_world_by_frame, corners_to_extent
    from bambi.io.dem import read_dem_mesh
    from bambi.io.poses import read_poses, to_local_poses
    from bambi.geo.poses import Poses

    root, trex, (mtx, dist), _, _, ref_geo, _ = data
    dem = read_dem_mesh(root / "qgis" / "flat_surface_dem.glb")
    pf = read_poses(root / "qgis" / "poses_w.json")
    poses = to_local_poses(pf, epsg=32643)
    und = undistort_boxes(trex.boxes, mtx, dist, trex.video_size, FRAME)
    corners = boxes_to_world_by_frame(trex.frames, und, poses, pf.fovy, FRAME[0], FRAME[1], dem.mesh, legacy=True)
    ext = corners_to_extent(corners) + np.tile(np.asarray(dem.metadata["origin"], float), 2)
    hit = np.isfinite(ext).all(axis=1)
    assert hit.sum() == len(ref_geo)                              # the plugin dropped the same misses
    assert np.array_equal(trex.frames[hit], ref_geo.frames) and np.array_equal(trex.track_ids[hit], ref_geo.track_ids)
    # This oracle was produced BEFORE the oblique-heading fix (2026-08-13): the
    # gimbal tilt of 0.1 deg entered the camera with the opposite sign, which at
    # 16.6 m altitude moves the ground point by 2 * tan(0.1 deg) * 16.6 = 5.8 cm
    # (up to 7 cm on the higher frames).
    # Everything else (undistortion, ray casting, extent, dropped misses) is exact,
    # so the tolerance is that known offset - not looseness.
    assert np.abs(ext[hit] - ref_geo.boxes).max() < 0.08
    flipped = Poses(positions=poses.positions, rotations=np.mod(poses.rotations * [-1, 1, 1], 360.0),
                    origin=poses.origin)
    corners = boxes_to_world_by_frame(trex.frames, und, flipped, pf.fovy, FRAME[0], FRAME[1], dem.mesh, legacy=True)
    ext = corners_to_extent(corners) + np.tile(np.asarray(dem.metadata["origin"], float), 2)
    assert np.abs(ext[hit] - ref_geo.boxes).max() < 0.02          # residual: old heading composition at 0.1 deg tilt
