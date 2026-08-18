# -*- coding: utf-8 -*-
"""Parity with the QGIS plugin's "Export Frames as GeoTIFF" on flight 146.

A miniature plugin project is built from the cached public flight (four
thermal frames cut from the processed video, a pipeline-schema
``poses_t.json``, the bundled DEM) and the plugin's real
``run_export_geotiffs`` is executed on it, qgis stubbed. Its GeoTIFFs are
then compared with :func:`bambi.render.ortho.frame_orthophoto`. Needs the
plugin checkout (``BAMBI_PLUGIN_REPO``) and the flight cache (slow tier).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("alfspy")
pytest.importorskip("rasterio")
cv2 = pytest.importorskip("cv2")

from tests.plugin_oracle import plugin_processor, requires_plugin  # noqa: E402

pytestmark = [pytest.mark.slow, requires_plugin]

FRAMES = [1900, 2400, 2900, 3400]
W = H = 1024
FOVY = 50.0


@pytest.fixture(scope="module")
def project(tmp_path_factory, data_dir):
    from bambi.io.dem import geotiff_to_dem, read_dem_mesh
    from bambi.io.poses import read_poses, to_local_poses, write_poses

    flight = data_dir / "base" / "146"
    video = flight / "146_matched_processed.mp4"
    poses_json = flight / "146_matched_poses.json"
    if not video.exists() or not poses_json.exists():
        pytest.skip("flight 146 (base) not cached")
    root = tmp_path_factory.mktemp("proj146")
    dem_dir = flight / "DEM"
    glb = dem_dir / "146_matched_dem.glb"
    if not glb.exists():
        fixture = Path(__file__).parent / "fixtures" / "dem" / "146_matched_dem.tif"
        dem_dir.mkdir(exist_ok=True)
        geotiff_to_dem(fixture, glb, simplify=2)
    dem = read_dem_mesh(glb)
    pf = read_poses(poses_json)
    poses = to_local_poses(pf, epsg=dem.origin.epsg, origin=dem.origin)
    frames_dir = root / "frames_t"
    frames_dir.mkdir()
    cap = cv2.VideoCapture(str(video))
    # every pose gets a file name (as the extractors write them); only FRAMES exist on disk
    names = [f"{i}-{i}-{i}.jpg" for i in range(len(poses))]
    for f in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok
        cv2.imwrite(str(frames_dir / names[f]), img[:, :W])
    cap.release()
    write_poses(root / "poses_t.json", poses, names, fovy=np.full(len(poses), FOVY), lla=pf.lla,
                timestamps=pf.timestamps, header={"drone": "DJI_M3T", "camera": "Thermal"})
    corr = json.loads((flight / "146_correction.json").read_text())
    return {"root": root, "glb": glb, "dem": dem, "poses": poses, "frames_dir": frames_dir, "names": names,
            "correction": corr}


def test_frame_geotiffs_match_the_plugin(project):
    from bambi.io.corrections import corrections_for_frames, parse_corrections
    from bambi.io.dem import read_render_data
    from bambi.io.geotiff import read_geotiff
    from bambi.render.masks import default_mask_polygon
    from bambi.render.ortho import frame_orthophoto
    from bambi.util.render_context import make_render_context

    proc = plugin_processor()
    root = project["root"]
    corr = project["correction"]
    config = {"target_folder": str(root), "dem_path": str(project["glb"]), "geotiff_camera": "T",
              "target_epsg": project["dem"].origin.epsg, "alfs_use_all_frames": False,
              "alfs_start_frame": min(FRAMES), "alfs_end_frame": max(FRAMES), "alfs_frame_step": 1,
              "alfs_ground_resolution": 0.1, "geotiff_edge_erosion_px": 2,
              "translation": corr["translation"], "rotation": corr["rotation"],
              "additional_corrections": [{"start": e["start_frame"], "end": e["end_frame"],
                                          "translation": e["translation"], "rotation": e["rotation"]}
                                         for e in corr.get("fine_corrections", [])]}
    logs = []
    proc.run_export_geotiffs(config, log_fn=logs.append)
    out_dir = root / "geotiffs_t"
    written = sorted(out_dir.glob("*.tiff"))
    assert [int(p.stem) for p in written] == FRAMES, logs[-5:]

    poses = project["poses"]
    tc, rc = corrections_for_frames(parse_corrections(corr), len(poses))
    mesh_data, texture_data = read_render_data(project["glb"])
    ctx = make_render_context()
    origin = np.asarray(project["dem"].metadata["origin"], float)
    poly = default_mask_polygon(W, H)
    for f, tif in zip(FRAMES, written):
        ref_img, ref_bounds, ref_epsg, _ = read_geotiff(tif)
        image = cv2.imread(str(project["frames_dir"] / project["names"][f]))
        for convention, tol in (("zyx", 0.98), ("drone_pose", 0.6)):
            res = frame_orthophoto(ctx, mesh_data, texture_data, project["dem"].mesh, image, poses.positions[f],
                                   poses.rotations[f], FOVY, W, H, polygon=poly, ground_resolution=0.1,
                                   translation_correction=tc[f], rotation_correction=rc[f], edge_erosion_px=2,
                                   convention=convention)
            assert res is not None
            img, valid, bounds = res
            world = (bounds[0] + origin[0], bounds[1] + origin[1], bounds[2] + origin[0], bounds[3] + origin[1])
            assert np.allclose(world, ref_bounds, atol=1e-4), (f, world, ref_bounds)
            assert img.shape == ref_img.shape, (img.shape, ref_img.shape)
            ours = img.copy()
            ours[~valid] = 0
            diff = np.abs(ours.astype(int) - ref_img.astype(int))
            # 'zyx' is what the plugin builds its shots with -> the same picture. The engine's
            # default (quaternion_from_drone_pose) tilts the other way by this flight's 0.1 deg,
            # i.e. ~1.5 px of texture shift at 0.1 m/px: same footprint, softer pixel agreement.
            assert (diff <= 2).mean() > tol, (f, convention, (diff <= 2).mean())
            assert ((ours.max(axis=2) > 0) == (ref_img.max(axis=2) > 0)).mean() > (0.995 if convention == "zyx" else 0.98)
