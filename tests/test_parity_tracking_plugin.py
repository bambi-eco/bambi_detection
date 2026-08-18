# -*- coding: utf-8 -*-
"""Parity with the QGIS plugin's built-in tracker, run for real.

The plugin's ``BambiProcessor._run_builtin_tracking`` (Hungarian IoU +
gap interpolation) is executed against its own project store - qgis stubbed
the way the plugin's unit tier does - on the lion flight's geo-referenced
detections, and its ``tracks.csv`` is compared row for row with
:mod:`bambi.tracking.iou`. Needs the plugin checkout and the private flight::

    BAMBI_PLUGIN_REPO=C:/D/Projects/Bambi-QGIS
    BAMBI_PARITY_GEOREF=C:/.../qgis_fixed      (georeferenced_t/georeferenced.txt)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN = os.environ.get("BAMBI_PLUGIN_REPO")
ROOT = os.environ.get("BAMBI_PARITY_GEOREF")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (PLUGIN and ROOT), reason="set BAMBI_PLUGIN_REPO and BAMBI_PARITY_GEOREF")]


@pytest.fixture(scope="module")
def plugin():
    repo = Path(PLUGIN)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import importlib.util
    spec = importlib.util.spec_from_file_location("bambi_qgis_stub", repo / "tests" / "qgis_stub.py")
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)                             # the plugin's own stub
    stub.install_qgis_stubs()
    from bambi_wildlife_detection.bambi_processing import BambiProcessor
    from bambi_wildlife_detection.core import detection_store, store, track_store
    return BambiProcessor(), detection_store, store, track_store


def _limit():
    return int(os.environ.get("BAMBI_PARITY_LIMIT", "4000")) or None


@pytest.mark.parametrize("mode", ["HUNGARIAN"])
def test_builtin_tracker_and_interpolation_match_row_for_row(plugin, tmp_path, mode):
    from bambi.io import tracks as io
    from bambi.tracking.iou import interpolate_tracks, track_boxes

    processor, detection_store, store, track_store = plugin
    geo = io.read_georeferenced(Path(ROOT) / "georeferenced_t" / "georeferenced.txt")
    n = _limit() or len(geo)
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": int(f), "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "confidence": float(c),
         "source_class": str(int(k))}
        for f, c, k in zip(geo.frames[:n], geo.confidences[:n], geo.classes[:n])])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    assert len(ids) == n
    track_store.record_georeference(root, "t", [
        {"detection_id": i, "gx1": b[0], "gy1": b[1], "gz1": b[2], "gx2": b[3], "gy2": b[4], "gz2": b[5]}
        for i, b in zip(ids, geo.boxes[:n])])
    processor._run_builtin_tracking({"target_folder": root, "tracking_camera": "T", "tracker_mode": mode,
                                     "iou_threshold": 0.3, "class_aware": True, "interpolate": True,
                                     "max_age": -1, "write_legacy_text_outputs": True})
    ref = io.read_tracks_csv(Path(root) / "tracks_t" / "tracks.csv")

    # the store hands the tracker rows in detection_id order == file order
    tid = track_boxes(geo.frames[:n], geo.boxes[:n], geo.classes[:n], mode=mode.lower(), iou_threshold=0.3,
                      class_aware=True, max_age=-1)
    ours = interpolate_tracks(geo.frames[:n], tid, geo.boxes[:n], geo.confidences[:n], geo.classes[:n],
                              confidence="mean")
    assert len(ours) == len(ref), (len(ours), len(ref))
    assert np.array_equal(ours.frames, ref.frames) and np.array_equal(ours.track_ids, ref.track_ids)
    assert np.array_equal(ours.interpolated, ref.interpolated) and np.array_equal(ours.classes, ref.classes)
    assert np.abs(ours.boxes - ref.boxes).max() < 1e-6                     # file has 6 decimals
    assert np.abs(ours.confidences - ref.confidences).max() < 1e-6
    written = io.write_tracks_csv(tmp_path / "ours.csv", ours)
    # the plugin writes platform line endings; the engine always writes LF
    assert written.read_text() == (Path(root) / "tracks_t" / "tracks.csv").read_text()
