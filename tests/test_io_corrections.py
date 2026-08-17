# -*- coding: utf-8 -*-
"""bambi.io.corrections - four dialects, one resolution rule."""
import json

import numpy as np

from bambi.io.corrections import Corrections, corrections_for_frames, parse_corrections, read_corrections


def _check(c):
    t, r = corrections_for_frames(c, 6)
    assert np.allclose(t[:2], [[0, 0, 0.2]] * 2) and np.allclose(r[:2], [[0, 0, -0.01]] * 2)
    assert np.allclose(t[2:5], [[0, 0, -1.0]] * 3) and np.allclose(r[2:5], [[0, 0, -0.5]] * 3)
    assert np.allclose(t[5], [0, 0, 0.2]) and np.allclose(r[5], [0, 0, -0.01])


def test_plugin_dialect():
    _check(parse_corrections({"translation": {"x": 0, "y": 0, "z": 0.2}, "rotation": {"x": 0, "y": 0, "z": -0.01},
                              "additional": [{"start": 2, "end": 4, "translation": {"z": -1.0},
                                              "rotation": {"z": -0.5}}]}))


def test_pipeline_config_dialect():
    _check(parse_corrections({"translation": {"z": 0.2}, "rotation": {"z": -0.01},
                              "additional_corrections": [{"start": 2, "end": 4, "translation": {"z": -1.0},
                                                          "rotation": {"z": -0.5}}]}))


def test_alfspy_dialect():
    _check(parse_corrections({"default": {"translation": {"z": 0.2}, "rotation": {"z": -0.01}},
                              "corrections": [{"start frame": 2, "end frame": 4, "translation": {"z": -1.0},
                                               "rotation": {"z": -0.5}}]}))


def test_public_dataset_dialect(tmp_path):
    data = {"rotation": {"x": 0.0, "y": 0.0, "z": -0.01}, "translation": {"x": 0.0, "y": 0.0, "z": 0.2},
            "fine_corrections": [{"start_frame": 2, "end_frame": 4, "rotation": {"x": 0.0, "y": 0.0, "z": -0.5},
                                  "translation": {"x": 0.0, "y": 0.0, "z": -1.0}}]}
    p = tmp_path / "146_correction.json"; p.write_text(json.dumps(data))
    _check(read_corrections(p))


def test_first_matching_range_wins_and_open_end():
    c = parse_corrections({"additional": [{"start": 0, "end": 3, "translation": {"x": 1}},
                                          {"start": 2, "translation": {"x": 2}}]})
    t, _ = corrections_for_frames(c, 6)
    assert np.allclose(t[:, 0], [1, 1, 1, 1, 2, 2])


def test_missing_file_is_identity(tmp_path):
    c = read_corrections(tmp_path / "nope.json")
    assert c == Corrections() and c.is_identity
    t, r = corrections_for_frames(c, 3)
    assert not t.any() and not r.any()
