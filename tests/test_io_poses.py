# -*- coding: utf-8 -*-
"""bambi.io.poses - both poses-file schemas in and out, as arrays."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")

from bambi.geo.poses import geographic_to_local, make_origin  # noqa: E402
from bambi.io.poses import read_poses, to_local_poses, write_poses  # noqa: E402

ORIGIN = {"latitude": 46.6448, "longitude": 14.5593, "altitude": 450.0}


def _pipeline_file(tmp_path):
    data = {
        "origin": ORIGIN, "drone": "DJI_M3T", "camera": "Thermal", "mask": "mask_T.png",
        "images": [
            {"imagefile": "0-0-0.jpg", "location": [10.0, 20.0, 30.0], "rotation": [0.0, 0, 51.6],
             "fovy": [46.375], "lat": 46.6451, "lng": 14.5605, "timestamp": "2023-01-19T11:39:00+01:00"},
            {"imagefile": "1-1-1.jpg", "location": [11.0, 21.0, 31.0], "rotation": [76.4, 0, 52.4],
             "fovy": [46.375], "lat": 46.6452, "lng": 14.5606, "timestamp": "2023-01-19T11:39:01+01:00"},
        ],
    }
    p = tmp_path / "poses_t.json"; p.write_text(json.dumps(data), encoding="utf-8"); return p


def _geographic_file(tmp_path):
    data = {
        "origin": ORIGIN, "drone": "DJI_M3T", "camera": "Thermal",
        "images": [
            {"lat": 46.64515, "lng": 14.56054, "alt": 479.5, "pitch": 360.1, "roll": 0, "yaw": 1.55,
             "timestamp": "2023-01-19T11:39:00.906000+01:00"},
            {"lat": 46.64520, "lng": 14.56060, "alt": 480.0, "pitch": 45.0, "roll": 0, "yaw": 90.0,
             "timestamp": "2023-01-19T11:39:01.906000+01:00"},
        ],
    }
    p = tmp_path / "146_matched_poses.json"; p.write_text(json.dumps(data), encoding="utf-8"); return p


# ---------------------------------------------------------------- reading

def test_reads_pipeline_schema(tmp_path):
    pf = read_poses(_pipeline_file(tmp_path))
    assert pf.schema == "pipeline" and len(pf) == 2
    assert pf.positions.shape == (2, 3) and pf.rotations.shape == (2, 3)
    assert np.allclose(pf.positions[1], [11, 21, 31])
    assert np.allclose(pf.fovy, [46.375, 46.375])
    assert pf.imagefiles == ["0-0-0.jpg", "1-1-1.jpg"]
    assert pf.lla[0, 0] == 46.6451 and np.isnan(pf.lla[0, 2])   # no per-image alt in this schema
    assert pf.header["drone"] == "DJI_M3T" and "images" not in pf.header


def test_reads_geographic_schema(tmp_path):
    pf = read_poses(_geographic_file(tmp_path))
    assert pf.schema == "geographic"
    assert pf.positions is None
    assert pf.lla.shape == (2, 3) and pf.rotations.shape == (2, 3)
    # public pitch IS the pose tilt: 360.1 wraps to 0.1 (nadir), 45 stays 45
    assert np.allclose(pf.rotations[0], [0.1, 0.0, 1.55])
    assert np.allclose(pf.rotations[1], [45.0, 0.0, 90.0])
    assert np.all(np.isnan(pf.fovy))
    assert pf.imagefiles == ["", ""]


# ---------------------------------------------------------------- to local

def test_pipeline_file_is_already_local(tmp_path):
    pf = read_poses(_pipeline_file(tmp_path))
    poses = to_local_poses(pf, epsg=32633)
    assert np.allclose(poses.positions, pf.positions)
    assert np.allclose(poses.rotations, pf.rotations)
    assert poses.origin.latitude == ORIGIN["latitude"]


def test_geographic_file_is_converted_like_the_dataset_tooling(tmp_path):
    """Positions from lat/lng/alt; rotation = [pitch, roll, yaw] verbatim (mod 360)."""
    pf = read_poses(_geographic_file(tmp_path))
    poses = to_local_poses(pf, epsg=32633)
    o = make_origin(ORIGIN["latitude"], ORIGIN["longitude"], ORIGIN["altitude"], 32633)
    assert np.allclose(poses.positions, geographic_to_local(pf.lla, o))
    assert np.allclose(poses.rotations, [[0.1, 0.0, 1.55], [45.0, 0.0, 90.0]])
    assert poses.positions[0, 2] == pytest.approx(479.5 - 450.0)
    # NOT gimbal_to_rotation: that would add 90 to an angle that already had it
    assert not np.allclose(poses.rotations[1, 0], 135.0)


def test_missing_origin_needs_an_explicit_one(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"images": [{"lat": 1, "lng": 2, "alt": 3, "pitch": 0, "roll": 0, "yaw": 0}]}))
    with pytest.raises(ValueError, match="origin"):
        to_local_poses(read_poses(p), epsg=32633)


# ---------------------------------------------------------------- writing

def test_write_then_read_round_trips(tmp_path):
    src = read_poses(_pipeline_file(tmp_path))
    poses = to_local_poses(src, epsg=32633)
    out = tmp_path / "out.json"
    write_poses(out, poses, src.imagefiles, fovy=src.fovy, lla=src.lla,
                timestamps=src.timestamps, header={"drone": "DJI_M3T", "camera": "Thermal"})
    back = read_poses(out)
    assert back.schema == "pipeline"
    assert np.allclose(back.positions, src.positions)
    assert np.allclose(back.rotations, src.rotations)
    assert np.allclose(back.fovy, src.fovy)
    assert back.imagefiles == src.imagefiles and back.timestamps == src.timestamps
    assert back.origin == ORIGIN and back.header["drone"] == "DJI_M3T"
    # written in the shape the extractors use: fovy as a one-element list
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["images"][0]["fovy"] == [46.375]


def test_write_rejects_length_mismatch(tmp_path):
    src = read_poses(_pipeline_file(tmp_path))
    poses = to_local_poses(src, epsg=32633)
    with pytest.raises(ValueError):
        write_poses(tmp_path / "bad.json", poses, ["only-one.jpg"])
