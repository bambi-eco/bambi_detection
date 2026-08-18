# -*- coding: utf-8 -*-
"""bambi.tracking.matching - cross-modal track matching (paper section 3.2).

Ported from the QGIS plugin's ``tests/test_track_matching.py``: the affine is
bootstrapped from data, so every test is built around a known ground-truth
transform - synthetic RGB detections are pushed through it to make the
thermal side, and the matcher has to recover both the transform and which
track is which. Frame correspondence by time is added.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from bambi.tracking import matching as tm  # noqa: E402
from bambi.tracking.matching import Affine, Candidates, MatchConfig  # noqa: E402

# A mild zoom, a small rotation and an offset - the kind of relationship two
# lenses on one airframe have.
TRUTH = Affine.from_coefficients(0.80, 0.05, -0.05, 0.80, 30.0, -12.0)


def _track(track_id, frames, x0, y0, dx=4.0, dy=2.0, confidence=0.8, first_id=0, curve=6.0):
    """A track of boxes as arrays. Not a straight line: an affine is
    unconstrained perpendicular to a line."""
    i = np.arange(len(frames), dtype=float)
    cx = x0 + dx * i + curve * np.sin(i * 0.7)
    cy = y0 + dy * i + curve * np.cos(i * 0.9)
    return dict(frames=np.asarray(frames), track_ids=np.full(len(i), track_id), boxes=np.column_stack(
        [cx - 5, cy - 5, cx + 5, cy + 5]), confidences=np.full(len(i), confidence),
        detection_ids=first_id + np.arange(len(i)))


def _cat(*tracks):
    return {k: np.concatenate([t[k] for t in tracks]) for k in tracks[0]}


def _through(rows, affine=TRUTH, first_id=1000, jitter=0.0):
    """Map RGB rows onto the thermal side using *affine*."""
    c = 0.5 * (rows["boxes"][:, :2] + rows["boxes"][:, 2:])
    m = affine.apply(c)
    i = np.arange(len(m))
    m[:, 0] += jitter * ((i % 3) - 1)
    m[:, 1] += jitter * ((i % 2) - 0.5)
    return dict(rows, boxes=np.column_stack([m[:, 0] - 4, m[:, 1] - 4, m[:, 0] + 4, m[:, 1] + 4]),
                detection_ids=first_id + i)


def _det(rows):
    return tm.detections(rows["frames"], rows["track_ids"], rows["boxes"], rows["confidences"], rows["detection_ids"])


def _identity(frames):
    f = np.asarray(list(frames))
    return np.column_stack([f, f])


# ---------------------------------------------------------------------------
# Frame correspondence
# ---------------------------------------------------------------------------

class TestFrameMatching:
    def test_nearest_within_tolerance(self):
        src = [0.0, 1.0, 2.0, 5.0, np.nan]
        dst = [0.02, 0.98, 3.0, np.nan]
        got = tm.match_frames_by_time(src, dst, max_dt=0.1)
        assert got.tolist() == [0, 1, -1, -1, -1]

    def test_ties_take_the_earlier_frame_and_unsorted_dst_is_fine(self):
        got = tm.match_frames_by_time([1.0], [1.5, 0.5], max_dt=1.0)
        assert got.tolist() == [1]                       # index of 0.5 (earlier in time)
        assert tm.match_frames_by_time([1.0], [], 1.0).tolist() == [-1]

    def test_frame_pairs_from_map(self):
        assert tm.frame_pairs_from_map([3, -1, 4]).tolist() == [[0, 3], [2, 4]]


# ---------------------------------------------------------------------------
# Fitting the affine
# ---------------------------------------------------------------------------

class TestFitAffine:
    def test_recovers_a_known_transform_exactly(self):
        src = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [37, 61]], float)
        fit = tm.fit_affine(src, TRUTH.apply(src))
        assert np.allclose(fit.matrix, TRUTH.matrix, atol=1e-9) and np.allclose(fit.offset, TRUTH.offset, atol=1e-7)

    def test_is_robust_to_small_noise(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(0, 500, size=(40, 2))
        fit = tm.fit_affine(src, TRUTH.apply(src) + rng.normal(0, 0.3, size=(40, 2)))
        assert np.allclose(fit.matrix, TRUTH.matrix, atol=0.01)
        assert tm.affine_rmse(fit, src, TRUTH.apply(src)) < 0.5

    def test_too_few_collinear_or_coincident_points_are_unfittable(self):
        assert tm.fit_affine([[0, 0], [1, 1]], [[0, 0], [1, 1]]) is None
        line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
        assert tm.fit_affine(line, TRUTH.apply(line)) is None
        assert tm.fit_affine(np.ones((5, 2)), np.ones((5, 2))) is None

    def test_rmse(self):
        src = np.array([[0, 0], [10, 0], [0, 10]], float)
        assert tm.affine_rmse(TRUTH, src, TRUTH.apply(src)) == pytest.approx(0.0)
        assert tm.affine_rmse(TRUTH, np.zeros((0, 2)), np.zeros((0, 2))) == float("inf")

    def test_list_form_round_trips_and_helpers(self):
        assert Affine.from_list(TRUTH.to_list()) == TRUTH
        assert Affine.from_list(None) == Affine.identity()
        assert TRUTH.coefficients == pytest.approx((0.80, 0.05, -0.05, 0.80, 30.0, -12.0))
        sx, sy = Affine.scaling(0.5, 0.25).inverse_scale()
        assert (sx, sy) == (2.0, 4.0)


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

class TestEstimateAffine:
    def test_seeds_only_from_unambiguous_frames(self):
        rgb = _cat(_track(1, [0, 1, 2, 3], 100.0, 100.0), _track(2, [3], 400.0, 400.0, first_id=50))
        src, dst = tm.seed_pairs(_det(_through(rgb)), _det(rgb), _identity(range(4)))
        assert len(src) == 3 == len(dst)

    def test_recovers_the_transform_from_a_flight(self):
        rgb = _cat(_track(1, range(12), 100.0, 120.0), _track(2, range(12), 500.0, 300.0, dx=-3.0, dy=5.0, first_id=100))
        fit = tm.estimate_affine(_det(_through(rgb)), _det(rgb), _identity(range(12)))
        assert fit.rmse < 0.01 and fit.n_pairs >= 12
        assert fit.affine.matrix[0, 0] == pytest.approx(TRUTH.matrix[0, 0], abs=0.01)
        assert fit.affine.offset[1] == pytest.approx(TRUTH.offset[1], abs=0.5)

    def test_a_herd_with_no_unambiguous_frame_still_registers(self):
        rgb = _cat(_track(1, range(10), 100.0, 120.0), _track(2, range(10), 600.0, 400.0, dx=-2.0, dy=3.0, first_id=100),
                   _track(3, range(10), 300.0, 700.0, dx=1.0, dy=-4.0, first_id=200))
        det_t, det_w = _det(_through(rgb)), _det(rgb)
        assert len(tm.seed_pairs(det_t, det_w, _identity(range(10)))[0]) == 0
        fit = tm.estimate_affine(det_t, det_w, _identity(range(10)), frame_size_t=(1024, 1024), frame_size_w=(1024, 1024))
        assert fit.rmse < 0.01 and fit.n_pairs == 30 and not fit.seeded
        assert fit.affine.matrix[0, 0] == pytest.approx(TRUTH.matrix[0, 0], abs=0.01)

    def test_one_perfectly_straight_track_cannot_register(self):
        rgb = _track(1, range(20), 100.0, 120.0, curve=0.0)
        fit = tm.estimate_affine(_det(_through(rgb)), _det(rgb), _identity(range(20)))
        assert fit.rmse == float("inf") and fit.affine == Affine.identity()

    def test_a_second_track_is_enough_to_register_straight_paths(self):
        rgb = _cat(_track(1, range(20), 100.0, 120.0, dx=5.0, dy=1.0, curve=0.0),
                   _track(2, range(20), 600.0, 500.0, dx=-1.0, dy=6.0, curve=0.0, first_id=100))
        assert tm.estimate_affine(_det(_through(rgb)), _det(rgb), _identity(range(20))).rmse < 0.01

    def test_falls_back_to_frame_size(self):
        empty = tm.detections([], [], np.zeros((0, 4)))
        fit = tm.estimate_affine(empty, empty, np.zeros((0, 2), int), frame_size_t=(640, 512), frame_size_w=(1280, 1024))
        assert fit.affine.matrix[0, 0] == pytest.approx(0.5) and fit.affine.matrix[1, 1] == pytest.approx(0.5)
        assert fit.rmse == float("inf")
        assert tm.estimate_affine(empty, empty, np.zeros((0, 2), int)).affine == Affine.identity()


# ---------------------------------------------------------------------------
# Track costs
# ---------------------------------------------------------------------------

class TestCandidates:
    def test_median_distance_of_a_true_pair_is_near_zero(self):
        rgb = _track(1, range(10), 100.0, 120.0)
        found = tm.candidates(_det(_through(rgb)), _det(rgb), _identity(range(10)), TRUTH)
        assert len(found) == 1 and found.shared[0] == 10 and found.median_dist[0] == pytest.approx(0.0, abs=1e-6)

    def test_a_single_bad_frame_does_not_break_a_pair(self):
        rgb = _track(1, range(11), 100.0, 120.0)
        thermal = _through(rgb)
        thermal["boxes"][5, [0, 2]] += 900
        found = tm.candidates(_det(thermal), _det(rgb), _identity(range(11)), TRUTH)
        assert found.median_dist[0] < 1.0

    def test_every_co_occurring_pair_is_considered(self):
        rgb = _cat(_track(1, range(10), 100.0, 120.0), _track(2, range(10), 300.0, 200.0, first_id=100))
        found = tm.candidates(_det(_through(rgb)), _det(rgb), _identity(range(10)), TRUTH)
        assert len(found) == 4      # 2 x 2, the wrong pairings included

    def test_tracks_that_never_co_occur_are_not_candidates(self):
        rgb = _track(1, range(0, 5), 100.0, 120.0)
        thermal = _through(_track(2, range(20, 25), 100.0, 120.0), first_id=2000)
        assert len(tm.candidates(_det(thermal), _det(rgb), _identity(range(30)), TRUTH)) == 0

    def test_mean_confidence_is_per_track(self):
        rgb = _track(1, range(4), 100.0, 120.0, confidence=0.9)
        thermal = _through(rgb)
        thermal["confidences"] = np.full(4, 0.3)
        found = tm.candidates(_det(thermal), _det(rgb), _identity(range(4)), TRUTH)
        assert found.conf_w[0] == pytest.approx(0.9) and found.conf_t[0] == pytest.approx(0.3)

    def test_frame_correspondence_and_ids_are_carried_through(self):
        rgb = _track(1, [0, 1, 2], 100.0, 120.0)
        thermal = _through(rgb, first_id=7000)
        shifted = dict(rgb, frames=rgb["frames"] + 10)
        found = tm.candidates(_det(thermal), _det(shifted), np.array([[0, 10], [1, 11], [2, 12]]), TRUTH)
        rows = found.pairs_of(0)
        assert list(zip(found.pair_frame_t[rows], found.pair_frame_w[rows])) == [(0, 10), (1, 11), (2, 12)]
        assert found.pair_detection_t[rows][0] == 7000 and found.pair_detection_w[rows][0] == 0

    def test_sorted_closest_first(self):
        rgb = _cat(_track(1, range(10), 100.0, 120.0), _track(2, range(10), 300.0, 200.0, first_id=100))
        found = tm.candidates(_det(_through(rgb)), _det(rgb), _identity(range(10)), TRUTH)
        assert np.all(np.diff(found.median_dist) >= 0)


# ---------------------------------------------------------------------------
# Gates and assignment
# ---------------------------------------------------------------------------

def _cands(*rows):
    """rows of (track_t, track_w, shared, median, conf_t, conf_w)."""
    defaults = (1, 11, 10, 5.0, 0.5, 0.5)
    r = np.array([tuple(v if v is not None else d for v, d in zip(row + (None,) * (6 - len(row)), defaults))
                  for row in rows], dtype=float).reshape(-1, 6)
    e = np.zeros(0, np.int64)
    return Candidates(r[:, 0].astype(np.int64), r[:, 1].astype(np.int64), r[:, 2].astype(np.int64), r[:, 3],
                      r[:, 4], r[:, 5], e, e, e, e, e, np.zeros(0))


class TestGates:
    def test_gates(self):
        assert tm.passes_gates(_cands((1, 11))).tolist() == [True]
        assert tm.passes_gates(_cands((1, 11, 7), (1, 11, 8))).tolist() == [False, True]
        assert tm.passes_gates(_cands((1, 11, 10, 27.9), (1, 11, 10, 28.0))).tolist() == [True, False]
        assert tm.passes_gates(_cands((1, 11, 10, 50.0)), MatchConfig(gate_px=60.0)).tolist() == [True]
        assert tm.passes_gates(_cands((1, 11, 10, 5.0, 0.1), (1, 11, 10, 5.0, 0.5, 0.1))).tolist() == [False, False]

    def test_defaults_match_the_paper(self):
        c = MatchConfig()
        assert (c.min_shared, c.gate_px, c.min_confidence) == (8, 28.0, 0.20)

    def test_rejection_reasons_are_counted(self):
        counts = tm.rejection_reasons(_cands((1, 11, 2), (1, 11, 10, 99.0), (1, 11, 10, 5.0, 0.05), (1, 11)))
        assert counts == {"shared_frames": 1, "distance": 1, "confidence": 1}


class TestAssign:
    def test_assignment_is_one_to_one(self):
        c = _cands((1, 11, 10, 1.0), (1, 12, 10, 2.0), (2, 11, 10, 3.0), (2, 12, 10, 4.0))
        idx = tm.assign(c)
        assert len(idx) == 2 and set(c.track_id_t[idx]) == {1, 2} and set(c.track_id_w[idx]) == {11, 12}

    def test_the_global_optimum_beats_the_greedy_choice(self):
        c = _cands((1, 11, 10, 1.0), (1, 12, 10, 2.0), (2, 11, 10, 20.0), (2, 12, 10, 50.0))
        idx = tm.assign(c, MatchConfig(gate_px=100.0))
        assert set(zip(c.track_id_t[idx], c.track_id_w[idx])) == {(1, 12), (2, 11)}

    def test_gated_pairs_are_never_assigned(self):
        c = _cands((1, 11, 10, 1.0), (2, 12, 10, 999.0))
        assert tm.assign(c).tolist() == [0]

    def test_unmatched_left_out_nothing_admissible_and_ordering(self):
        assert len(tm.assign(_cands((1, 11, 10, 1.0), (1, 12, 10, 2.0)))) == 1
        assert tm.assign(_cands((1, 11, 1))).tolist() == [] and tm.assign(_cands()).tolist() == []
        c = _cands((1, 11, 10, 9.0), (2, 12, 10, 1.0))
        assert c.median_dist[tm.assign(c)].tolist() == [1.0, 9.0]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestMatchTracks:
    def test_a_two_animal_flight_matches_correctly(self):
        rgb = _cat(_track(1, range(15), 100.0, 120.0), _track(2, range(15), 600.0, 400.0, dx=-3.0, dy=2.0, first_id=100))
        r = tm.match_tracks(_det(_through(rgb, jitter=0.4)), _det(rgb), _identity(range(15)))
        assert r.affine_rmse < 2.0
        c = r.candidates
        assert set(zip(c.track_id_t[r.matches], c.track_id_w[r.matches])) == {(1, 1), (2, 2)}
        assert (c.shared[r.matches] == 15).all()

    def test_short_low_confidence_and_single_modality_tracks_are_not_confirmed(self):
        rgb = _track(1, range(4), 100.0, 120.0)
        r = tm.match_tracks(_det(_through(rgb)), _det(rgb), _identity(range(4)))
        assert len(r.matches) == 0 and r.n_rejected == 1
        rgb = _track(1, range(15), 100.0, 120.0, confidence=0.05)
        assert len(tm.match_tracks(_det(_through(rgb)), _det(rgb), _identity(range(15))).matches) == 0
        rgb = _track(1, range(15), 100.0, 120.0)
        both = _cat(rgb, _track(2, range(15), 800.0, 500.0, first_id=100))
        r = tm.match_tracks(_det(_through(rgb)), _det(both), _identity(range(15)))
        assert r.candidates.track_id_w[r.matches].tolist() == [1]

    def test_frames_without_a_counterpart_are_skipped(self):
        rgb = _track(1, range(15), 100.0, 120.0)
        r = tm.match_tracks(_det(_through(rgb)), _det(rgb), _identity(range(10)))
        assert r.candidates.shared[r.matches][0] == 10

    def test_an_empty_flight_matches_nothing(self):
        empty = tm.detections([], [], np.zeros((0, 4)))
        r = tm.match_tracks(empty, empty, np.zeros((0, 2), int))
        assert len(r.matches) == 0 and r.n_candidates == 0

    def test_matches_the_plugin_reference_on_a_herd(self):
        """Same synthetic herd, the plugin's implementation as oracle (when the checkout is present)."""
        import os
        repo = os.environ.get("BAMBI_PLUGIN_REPO")
        if not repo:
            pytest.skip("set BAMBI_PLUGIN_REPO to compare against the plugin's core.track_matching")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plugin_tm", os.path.join(repo, "bambi_wildlife_detection", "core", "track_matching.py"))
        ptm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ptm)
        rgb = _cat(_track(1, range(20), 100.0, 120.0), _track(2, range(20), 600.0, 400.0, dx=-2.0, dy=3.0, first_id=100),
                   _track(3, range(12), 300.0, 700.0, dx=1.0, dy=-4.0, first_id=200))
        thermal = _through(rgb, jitter=1.5)
        rows = lambda d: [{"detection_id": int(i), "track_id": int(t), "frame": int(f), "x1": b[0], "y1": b[1],
                           "x2": b[2], "y2": b[3], "confidence": float(c)}
                          for i, t, f, b, c in zip(d["detection_ids"], d["track_ids"], d["frames"], d["boxes"], d["confidences"])]
        ref = ptm.match_tracks(rows(thermal), rows(rgb), {f: f for f in range(20)}, frame_size_t=(1024, 1024),
                               frame_size_w=(1024, 1024))
        ours = tm.match_tracks(_det(thermal), _det(rgb), _identity(range(20)), frame_size_t=(1024, 1024),
                               frame_size_w=(1024, 1024))
        assert ours.affine_rmse == pytest.approx(ref["affine_rmse"], abs=1e-9)
        assert np.allclose(ours.affine.coefficients, tuple(ref["affine"]), atol=1e-9)
        c = ours.candidates
        got = sorted((int(c.track_id_t[i]), int(c.track_id_w[i]), int(c.shared[i]), round(float(c.median_dist[i]), 9))
                     for i in ours.matches)
        want = sorted((m["track_id_t"], m["track_id_w"], m["shared"], round(m["median_dist"], 9)) for m in ref["matches"])
        assert got == want and ours.n_candidates == ref["candidates"]


class TestSeedTrap:
    def test_an_exact_three_point_seed_does_not_freeze_the_fit(self):
        """Three unambiguous frames fit an affine exactly (RMSE 0) - and a
        noisy one at that. Judging the refit on the same, larger pair set
        lets the herd's evidence overrule the seed."""
        rng = np.random.default_rng(5)
        rgb = _cat(_track(1, range(3), 100.0, 120.0),                       # 3 lone frames -> the seed
                   _track(1, range(3, 30), 112.0, 126.0, first_id=10),
                   _track(2, range(3, 30), 600.0, 400.0, dx=-2.0, dy=3.0, first_id=100),
                   _track(3, range(3, 30), 300.0, 700.0, dx=1.0, dy=-4.0, first_id=200))
        thermal = _through(rgb)
        thermal["boxes"][:3] += rng.normal(0, 3.0, size=(3, 4))             # jitter only the seed frames
        fit = tm.estimate_affine(_det(thermal), _det(rgb), _identity(range(30)))
        assert not fit.seeded and fit.n_pairs > 3        # the scale start won over the bad seed
        assert np.allclose(fit.affine.matrix, TRUTH.matrix, atol=0.01)
        r = tm.match_tracks(_det(thermal), _det(rgb), _identity(range(30)))
        assert len(r.matches) == 3
