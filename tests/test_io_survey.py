# -*- coding: utf-8 -*-
"""bambi.io.survey - survey files in the plugin's layouts, round-tripped."""
import json

import numpy as np
import pytest

from bambi.io import survey as io


def test_route_geojson_round_trip(tmp_path):
    coords = np.array([[500000.0, 5300000.0, 400.0], [500010.0, 5300005.0, 401.0]])
    p = io.write_route_geojson(tmp_path / "flight_route.geojson", coords, 32633)
    data = json.loads(p.read_text())
    assert data["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::32633"
    assert data["features"][0]["geometry"]["type"] == "LineString"
    assert data["features"][0]["properties"]["total_gps_points"] == 2
    assert np.array_equal(io.read_route_geojson(p), coords)
    single = io.write_route_geojson(tmp_path / "one.geojson", coords[:1], 32633)
    assert json.loads(single.read_text())["features"][0]["geometry"]["type"] == "Point"
    assert io.read_route_geojson(single).shape == (1, 3)
    pts = io.write_points_geojson(tmp_path / "cams.geojson", coords[:, :2], 32633, [{"frame": 0}, {"frame": 1}])
    assert json.loads(pts.read_text())["features"][1]["properties"] == {"frame": 1}
    (tmp_path / "empty.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    with pytest.raises(ValueError):
        io.read_route_geojson(tmp_path / "empty.geojson")


def test_fov_polygons_round_trip(tmp_path):
    polys = {0: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], float), 3: np.zeros((0, 3)),
             1: np.array([[1, 2, 3], [np.nan, 0, 0], [4, 5, 6], [7, 8, 9]])}
    p = io.write_fov_polygons(tmp_path / "fov_polygons.txt", polys)
    lines = p.read_text().splitlines()
    assert lines[0].startswith("# FoV polygon") and lines[2].startswith("0 4 1.000000 2.000000 3.000000")
    assert lines[3].startswith("1 3 ") and lines[4] == "3 0"
    back = io.read_fov_polygons(p)
    assert set(back) == {0, 1, 3} and back[0].shape == (4, 3) and back[3].shape == (0, 3)
    assert np.allclose(back[1], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_perpendicular_json_round_trip(tmp_path):
    frames = [5, 9]
    centres = [[100.0, 200.0, 30.0], [110.0, 210.0, 31.0]]
    feet = [[100.0, 190.0], [105.0, 210.0]]
    p = io.write_perpendicular_json(tmp_path / "perpendicular_t.json", frames, centres, feet, [10.0, 5.00004],
                                    [0.9, 0.8], [1, 2], 32633, ids=[7, 8])
    data = json.loads(p.read_text())
    assert data["crs"] == "EPSG:32633" and data["total_detections"] == 2
    assert data["perpendiculas"][0] == {"det_idx": 7, "frame": 5, "confidence": 0.9, "class_id": 1,
                                        "detection_center": [100.0, 200.0, 30.0], "foot_point": [100.0, 190.0],
                                        "distance_m": 10.0}
    assert data["perpendiculas"][1]["distance_m"] == 5.0                    # rounded to 4 decimals
    fr, c, ft, d, cls = io.read_perpendicular_json(p)
    assert fr.tolist() == frames and np.allclose(c, centres) and np.allclose(ft, feet) and cls.tolist() == [1, 2]
    t = io.write_perpendicular_json(tmp_path / "perpendicular_tracks_t.json", frames, centres, feet, [1, 2],
                                    [1, 1], [0, 0], 32633, ids=[3, 4], kind="tracks")
    row = json.loads(t.read_text())["tracks"][0]
    assert row["track_id"] == 3 and row["last_frame"] == 5 and "frame" not in row
    fr2, *_ = io.read_perpendicular_json(t)
    assert fr2.tolist() == frames


def test_transects_round_trip(tmp_path):
    p = io.write_transects(tmp_path / "transects.json", [[0, 99], [100, 250]], "t", names=["A", ""],
                           lengths_m=[123.456, 80.0], timestamps=[f"2023-01-19T11:{i:02d}:00+01:00" for i in range(60)] * 5)
    data = json.loads(p.read_text())
    assert data["version"] == 1 and data["modality"] == "t"
    assert data["transects"][0]["length_m"] == 123.46 and data["transects"][0]["start_time"].startswith("2023")
    ids, ranges, names = io.read_transects(p)
    assert ids.tolist() == [1, 2] and ranges.tolist() == [[0, 99], [100, 250]] and names == ["A", ""]
    csv_lines = (tmp_path / "transects.csv").read_text().splitlines()
    assert csv_lines[0] == "id,name,start_frame,end_frame,start_time,end_time,length_m"
    assert csv_lines[1].startswith("1,A,0,99,")
