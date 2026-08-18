# -*- coding: utf-8 -*-
"""bambi.io.tracks - every pipeline text table in and out, byte-identical writers."""
import numpy as np
import pytest

from bambi.io import tracks as io
from bambi.tracking.iou import Tracks

DET = """# frame x1 y1 x2 y2 confidence class_id
0 143.53 499.81 217.68 523.64 0.7417 0
0 244.52 502.02 300.60 529.90 0.7162 1
3 10.00 20.00 30.00 40.00 0.5000 0
"""
GEO = """# idx frame min_x min_y min_z max_x max_y max_z confidence class_id
0 0 265293.969383 9999950.148597 1818.921712 265300.645114 9999998.842918 1819.426298 0.7162 0
1 2 265292.594273 9999972.632632 1818.245615 265322.310027 10000048.040112 1819.242995 0.6604 1
"""
TRK = """00001184,1,264996.784332,9999459.995719,1811.536735,264996.784332,9999459.995719,1811.536735,0.884900,0,0
00001185,1,264996.884332,9999459.995719,1811.536735,264996.984332,9999460.995719,1811.536735,0.884900,0,1
00001186,2,264997.384234,9999461.842199,1811.586230,264997.384234,9999461.842199,1811.586230,0.889100,3,0
"""
PIX_PLUGIN = """# frame,track_id,x1,y1,x2,y2,conf,cls,interpolated
1184,1,143.53,499.81,217.68,523.64,0.8849,0,0
1185,1,144.00,500.00,218.00,524.00,0.8849,0,1
"""
PIX_TREX = """00001184,7,143.530000,499.810000,217.680000,523.640000,0.884900,0,0
"""
MOT = """1889,2557,183,903,35,40,1.00,61,0.50,Sus scrofa (Wild boar),2,2,0
1893,2555,263,898,41,38,1.00,60,1.00,Sus scrofa (Wild boar),2,2,0
"""


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="")
    return p


def test_read_detections(tmp_path):
    t = io.read_detections(_w(tmp_path, "detections.txt", DET))
    assert len(t) == 3 and not t.is_geo and t.boxes.shape == (3, 4)
    assert t.frames.tolist() == [0, 0, 3] and t.classes.tolist() == [0, 1, 0]
    assert np.allclose(t.boxes[0], [143.53, 499.81, 217.68, 523.64]) and t.confidences[1] == 0.7162
    assert (t.track_ids == -1).all()


def test_read_georeferenced(tmp_path):
    t = io.read_georeferenced(_w(tmp_path, "georeferenced.txt", GEO))
    assert t.is_geo and t.frames.tolist() == [0, 2]
    assert t.boxes[1, 5] == 1819.242995 and t.confidences[0] == 0.7162 and t.classes[1] == 1


def test_read_tracks_csv(tmp_path):
    t = io.read_tracks_csv(_w(tmp_path, "tracks.csv", TRK))
    assert t.frames.tolist() == [1184, 1185, 1186] and t.track_ids.tolist() == [1, 1, 2]
    assert t.interpolated.tolist() == [False, True, False] and t.classes.tolist() == [0, 0, 3]


def test_read_pixel_tracks_both_styles(tmp_path):
    a = io.read_pixel_tracks(_w(tmp_path, "a.csv", PIX_PLUGIN))
    b = io.read_pixel_tracks(_w(tmp_path, "b.csv", PIX_TREX))
    assert a.frames.tolist() == [1184, 1185] and a.interpolated.tolist() == [False, True]
    assert b.frames.tolist() == [1184] and b.track_ids.tolist() == [7]
    assert np.allclose(a.boxes[0], b.boxes[0])


def test_read_mot_converts_to_xyxy_and_keeps_extras(tmp_path):
    t = io.read_mot(_w(tmp_path, "gt.txt", MOT))
    assert t.track_ids.tolist() == [2557, 2555]
    assert np.allclose(t.boxes[0], [183, 903, 218, 943])
    assert t.classes.tolist() == [61, 60] and t.confidences.tolist() == [1.0, 1.0]
    assert t.extra.shape == (2, 5) and t.extra[0, 1] == "Sus scrofa (Wild boar)"


def _tracks(geo=False):
    boxes = np.array([[1.0, 2.0, 3.0, 4.0], [5.5, 6.5, 7.5, 8.5]])
    if geo:
        boxes = np.array([[1, 2, 0.5, 3, 4, 0.75], [5.5, 6.5, 1.0, 7.5, 8.5, 1.5]])
    return Tracks(np.array([3, 4]), np.array([1, 1]), boxes, np.array([0.9, 0.95]), np.array([0, 2]),
                  np.array([False, True]), np.array([0, -1]))


def test_writers_reproduce_the_plugin_formats(tmp_path):
    p = io.write_tracks_csv(tmp_path / "tracks.csv", _tracks(geo=True))
    assert p.read_text() == ("00000003,1,1.000000,2.000000,0.500000,3.000000,4.000000,0.750000,0.900000,0,0\n"
                             "00000004,1,5.500000,6.500000,1.000000,7.500000,8.500000,1.500000,0.950000,2,1\n")
    p = io.write_pixel_tracks(tmp_path / "px.csv", _tracks())
    assert p.read_text() == ("# frame,track_id,x1,y1,x2,y2,conf,cls,interpolated\n"
                             "3,1,1.00,2.00,3.00,4.00,0.9000,0,0\n4,1,5.50,6.50,7.50,8.50,0.9500,2,1\n")
    p = io.write_pixel_tracks(tmp_path / "px2.csv", _tracks(), style="trex")
    assert p.read_text().splitlines()[0] == "00000003,1,1.000000,2.000000,3.000000,4.000000,0.900000,0,0"
    p = io.write_detections(tmp_path / "d.txt", [3], [[1.234, 2, 3, 4]], [0.75], [1])
    assert p.read_text() == "# frame x1 y1 x2 y2 confidence class_id\n3 1.23 2.00 3.00 4.00 0.7500 1\n"
    p = io.write_georeferenced(tmp_path / "g.txt", [3], [[1, 2, 3, 4, 5, 6]], [0.5], [0])
    assert p.read_text().splitlines()[1] == "0 3 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 0.5000 0"
    p = io.write_mot(tmp_path / "m.txt", _tracks())
    assert p.read_text().splitlines()[0] == "3,1,1.00,2.00,2.00,2.00,0.90,0,1.00"


def test_round_trips(tmp_path):
    for writer, reader, geo in ((io.write_tracks_csv, io.read_tracks_csv, True),
                                (io.write_pixel_tracks, io.read_pixel_tracks, False),
                                (io.write_mot, io.read_mot, False)):
        t = _tracks(geo)
        back = reader(writer(tmp_path / "x", t))
        assert back.frames.tolist() == t.frames.tolist() and back.track_ids.tolist() == t.track_ids.tolist()
        assert np.allclose(back.boxes, t.boxes) and back.classes.tolist() == t.classes.tolist()
    with pytest.raises(ValueError):
        io.write_tracks_csv(tmp_path / "bad", _tracks(geo=False))


def test_tracks_from_table_sorts_and_keeps_source(tmp_path):
    t = io.tracks_from_table(io.read_tracks_csv(_w(tmp_path, "t.csv", TRK)))
    assert isinstance(t, Tracks) and t.source.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        io.tracks_from_table(io.read_detections(_w(tmp_path, "d.txt", DET)))


def test_empty_files(tmp_path):
    t = io.read_detections(_w(tmp_path, "e.txt", "# frame x1 y1 x2 y2 confidence class_id\n"))
    assert len(t) == 0 and t.boxes.shape == (0, 4)
