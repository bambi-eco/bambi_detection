# -*- coding: utf-8 -*-
"""Parity with the QGIS plugin's ``georeferenced.txt`` on a real flight.

The plugin (untouched, the oracle) geo-referenced 48k detections of the
Ol Pejeta lion flight through ``build_camera`` + ``label_to_world_coordinates``.
This re-does it with :mod:`bambi.geo.camera` + :mod:`bambi.geo.georef` and
compares number for number. Runs only where that private data is available::

    BAMBI_PARITY_GEOREF=C:/.../qgis_fixed  BAMBI_PARITY_DEM=C:/.../qgis/fabdem_utm.glb

Layout expected under ``BAMBI_PARITY_GEOREF``: ``poses_t.json``,
``detections_t/detections.txt``, ``georeferenced_t/georeferenced.txt``,
``frames_t/`` (for the frame size).
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyrr")
pytest.importorskip("alfspy")
pytest.importorskip("trimesh")

ROOT = os.environ.get("BAMBI_PARITY_GEOREF")
DEM = os.environ.get("BAMBI_PARITY_DEM")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (ROOT and DEM), reason="set BAMBI_PARITY_GEOREF and BAMBI_PARITY_DEM")]


def _read_table(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln and not ln.startswith("#")]
    return np.array(rows, dtype=np.float64)


@pytest.fixture(scope="module")
def data():
    import cv2
    from bambi.io.dem import read_dem_mesh
    from bambi.io.poses import read_poses, to_local_poses

    root = Path(ROOT)
    dem = read_dem_mesh(DEM)
    pf = read_poses(root / "poses_t.json")
    poses = to_local_poses(pf, epsg=dem.origin.epsg if dem.origin else 32737)
    frame = next(p for p in (root / "frames_t").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    h, w = cv2.imread(str(frame)).shape[:2]
    dets = _read_table(root / "detections_t" / "detections.txt")          # frame x1 y1 x2 y2 conf cls
    ref = _read_table(root / "georeferenced_t" / "georeferenced.txt")     # idx frame min.. max.. conf cls
    return dem, poses, pf.fovy, (w, h), dets, ref


def _limit():
    return int(os.environ.get("BAMBI_PARITY_LIMIT", "0")) or None


def test_georeferenced_txt_is_reproduced(data):
    from bambi.geo import georef
    from bambi.geo.camera import cameras_from_poses

    dem, poses, fovy, (w, h), dets, ref = data
    offset = np.asarray(dem.metadata["origin"], dtype=np.float64)
    n = _limit() or len(dets)
    dets = dets[:n]

    frames = dets[:, 0].astype(int)
    cams = {}
    ours = []
    for row, fi in zip(dets, frames):
        if fi not in cams:
            cams[fi] = cameras_from_poses(poses, fovy, aspect_ratio=w / h, indices=[fi])[0]
        x1, y1, x2, y2 = row[1:5]
        pts = georef.pixels_to_world_legacy([x1, x2, x2, x1], [y1, y1, y2, y2], w, h, dem.mesh, cams[fi])
        if len(pts) == 0:
            continue
        pts = pts + offset
        ours.append([fi, *pts.min(axis=0), *pts.max(axis=0), row[5], row[6]])
    ours = np.array(ours)

    ref = ref[ref[:, 1] < frames.max() + 1] if _limit() else ref
    ref = ref[:len(ours)] if _limit() else ref
    assert len(ours) == len(ref), (len(ours), len(ref))
    assert np.array_equal(ours[:, 0], ref[:, 1])                          # same frames, same order
    # the oracle file is printed with 6 decimals: agree to that
    diff = np.abs(ours[:, 1:7] - ref[:, 2:8])
    assert diff.max() < 1e-5, diff.max()
    assert np.allclose(ours[:, 7], ref[:, 8], atol=1e-4) and np.array_equal(ours[:, 8], ref[:, 9])


def test_nan_aligned_variant_differs_only_by_the_int_truncation(data):
    """The new API keeps misses aligned and does not truncate pixels. Near the
    horizon one pixel is hundreds of metres of ground, so compare in pixel
    space: re-projecting both results must land within one pixel of each other,
    and the legacy result must sit at the truncated pixel."""
    from bambi.geo import georef
    from bambi.geo.camera import cameras_from_poses, world_to_pixel

    dem, poses, fovy, (w, h), dets, ref = data
    sample = dets[:: max(1, len(dets) // 400)]
    checked, worst = 0, 0.0
    for row in sample:
        fi = int(row[0])
        camera = cameras_from_poses(poses, fovy, aspect_ratio=w / h, indices=[fi])[0]
        corners = georef.boxes_to_world(row[1:5], camera, w, h, dem.mesh)[0]
        legacy = georef.pixels_to_world_legacy([row[1], row[3], row[3], row[1]], [row[2], row[2], row[4], row[4]],
                                               w, h, dem.mesh, camera)
        if len(legacy) < 4 or not np.all(np.isfinite(corners)):
            continue
        px_new = world_to_pixel(corners, camera, w, h)
        px_old = world_to_pixel(legacy, camera, w, h)
        want_new = np.array([[row[1], row[2]], [row[3], row[2]], [row[3], row[4]], [row[1], row[4]]])
        assert np.allclose(px_new, want_new, atol=1e-2)
        # The legacy path (alfspy include_misses=False) returns hits in trimesh's
        # order, NOT the input order - fine for a min/max box, a landmine for a
        # polygon. Compare as sets.
        srt = lambda a: a[np.lexsort((np.round(a[:, 1], 2), np.round(a[:, 0], 2)))]
        assert np.allclose(srt(px_old), srt(np.trunc(want_new)), atol=1e-2)
        worst = max(worst, float(np.abs(srt(px_new) - srt(px_old)).max()))
        checked += 1
    assert checked > 100 and worst < 1.0, (checked, worst)
