# -*- coding: utf-8 -*-
"""bambi.survey.population - assignment, monitored area, the three estimators.

Ported from the QGIS plugin's ``tests/test_population_core.py``, golden
glmmTMB values included, plus a direct comparison with the plugin's module.
"""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")

from bambi.survey import population as pop  # noqa: E402
from tests.plugin_oracle import plugin_module, requires_plugin  # noqa: E402

LINES = [np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([[0.0, 100.0], [100.0, 100.0]])]


def _square(x0, y0, size=10.0):
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]]


def _strips():
    from shapely.geometry import Polygon
    return [Polygon([(0, -10), (100, -10), (100, 10), (0, 10)]), Polygon([(0, 90), (100, 90), (100, 110), (0, 110)])]


# ---------------------------------------------------------------- assignment

class TestAssign:
    def test_nearest_transect_and_distances(self):
        a = pop.assign_to_transects([[50, 10], [50, 90]], LINES)
        assert a.transect.tolist() == [0, 1] and np.allclose(a.distance, [10, 10])
        assert np.allclose(a.distances[0], [10, 90]) and a.counts(2).tolist() == [1, 1]

    def test_truncation(self):
        a = pop.assign_to_transects([[50, 10], [50, 50]], LINES, truncation=20.0)
        assert a.transect.tolist() == [0, -1] and a.truncated.tolist() == [False, True]
        assert a.distance[1] == pytest.approx(50.0)
        z = pop.assign_to_transects([[50, 5000]], LINES, truncation=0.0)
        assert z.transect.tolist() == [1] and not z.truncated[0]

    def test_no_transects(self):
        a = pop.assign_to_transects([[0, 0]], [])
        assert a.transect.tolist() == [-1] and np.isnan(a.distance[0]) and a.counts(0).shape == (0,)

    def test_containment(self):
        inside = pop.points_in_geometries([[50, 30], [50, 8], [50, 10]], _strips())
        assert inside.tolist() == [[False, False], [True, False], [True, False]]      # boundary counts
        a = pop.assign_to_transects([[50, 30], [50, 8], [50, 10]], LINES, inside=inside)
        assert a.transect.tolist() == [-1, 0, 0] and a.outside.tolist() == [True, False, False]
        assert np.isnan(a.distance[0]) and a.nearest_distance[0] == pytest.approx(30.0)
        assert a.distance[1] == pytest.approx(8.0)

    def test_overlapping_footprints_go_to_the_nearest_centre_line(self):
        from shapely.geometry import Polygon
        both = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        a = pop.assign_to_transects([[50, 40]], LINES, inside=pop.points_in_geometries([[50, 40]], [both, both]))
        assert a.transect.tolist() == [0] and not a.outside[0]

    def test_truncation_inside_footprint_and_missing_footprint(self):
        inside = pop.points_in_geometries([[50, 8]], _strips())
        a = pop.assign_to_transects([[50, 8]], LINES, truncation=5.0, inside=inside)
        assert a.transect.tolist() == [-1] and a.truncated[0] and not a.outside[0]
        inside2 = pop.points_in_geometries([[50, 8]], [None, _strips()[1]])
        b = pop.assign_to_transects([[50, 8]], LINES, inside=inside2)
        assert b.transect.tolist() == [-1] and b.outside[0]


class TestMergedArea:
    def test_union_not_sum(self):
        area, geom = pop.merged_footprint_area([_square(0, 0), _square(5, 0)])
        assert area == pytest.approx(150.0) and geom is not None
        assert pop.merged_footprint_area([_square(0, 0), [[np.nan, np.nan]] * 4])[0] == pytest.approx(100.0)
        assert pop.merged_footprint_area([]) == (0.0, None)
        assert pop.merged_footprint_area([[[0, 0, 0], [1, 1, 0]]]) == (0.0, None)
        rings = pop.geometry_rings(geom)
        assert len(rings) == 1 and np.allclose(rings[0][0], rings[0][-1])
        assert pop.geometry_rings(None) == []


# ---------------------------------------------------------------- estimators

class TestNaiveAndBootstrap:
    def test_naive_matches_the_r_formula(self):
        r = pop.estimate_naive([3, 0, 5], [1.5, 2.0, 0.5])
        assert r.density_per_100ha == pytest.approx(8 / 4.0 * 100)
        assert pop.estimate_naive([1], [0.0]).error is not None

    def test_bootstrap_centres_on_naive_and_is_reproducible(self):
        counts = [3, 0, 5, 2, 0, 1, 4]
        areas = [1.5, 2.0, 0.5, 1.0, 1.2, 0.8, 1.1]
        b = pop.estimate_bootstrap(counts, areas, n_boot=2000, seed=1)
        assert b.density_per_100ha == pytest.approx(pop.estimate_naive(counts, areas).density_per_100ha, rel=0.1)
        assert b.ci95[0] < b.density_per_100ha < b.ci95[1] and b.se > 0
        assert pop.estimate_bootstrap(counts, areas, 200, 7) == pop.estimate_bootstrap(counts, areas, 200, 7)
        assert pop.estimate_bootstrap([1], [1.0]).error is not None


class TestZinb:
    def test_recovers_a_simulated_density(self):
        rng = np.random.default_rng(12345)
        n = 60
        areas = rng.uniform(0.5, 3.0, size=n)
        mu = np.exp(0.2 + 1.0 * areas)
        counts = rng.negative_binomial(5, 5 / (5 + mu))
        counts = np.where(rng.random(n) < 0.3, 0, counts)
        r = pop.estimate_zinb(counts, areas)
        assert r.error is None and r.converged
        assert r.density_per_100ha == pytest.approx(pop.estimate_naive(counts, areas).density_per_100ha, rel=0.25)
        assert r.se > 0 and r.ci95[0] < r.density_per_100ha < r.ci95[1]
        assert 0.0 <= r.zero_inflation_prob <= 1.0 and math.isfinite(r.aic)

    def test_needs_enough_transects_and_area_variance(self):
        assert "at least 4" in pop.estimate_zinb([1, 0, 2], [1.0, 1.0, 1.0]).error
        assert "same area" in pop.estimate_zinb([1, 0, 2, 3, 0], [1.0] * 5).error

    def test_no_excess_zeros_flags_the_zero_inflation_boundary(self):
        rng = np.random.default_rng(4)
        areas = rng.uniform(1.0, 3.0, size=40)
        counts = rng.integers(3, 12, size=40)
        r = pop.estimate_zinb(counts, areas)
        assert r.zero_inflation_at_boundary and r.zero_inflation_prob < 1e-6
        assert r.density_per_100ha is not None and "no excess zeros" in r.error

    def test_poisson_limit_flags_the_dispersion_boundary(self):
        rng = np.random.default_rng(5)
        areas = rng.uniform(1.0, 3.0, size=50)
        counts = rng.poisson(np.exp(0.3 + 0.5 * areas))
        counts = np.where(rng.random(50) < 0.4, 0, counts)
        r = pop.estimate_zinb(counts, areas)
        assert r.dispersion_at_boundary and r.density_per_100ha is not None
        assert "Poisson limit" in r.error and r.se is not None and r.se > 0


class TestZinbAgainstGlmmTMB:
    """Golden values from the R analysis this module ports (Praschl et al. 2026,
    data/Flug1XDaim, An-/Rueckflug excluded): the dens_zinb / se_zinb that
    glmmTMB(count ~ ha, ziformula = ~1, family = nbinom2) produced."""

    LEMBACH_COUNTS = [0] * 23 + [1, 1, 1, 1, 2, 3, 3, 3, 5, 5]
    LEMBACH_AREAS = [1.475133, 1.521986, 1.575163, 1.577327, 1.593400, 1.679427, 1.696479, 1.698978, 1.763420,
                     1.827896, 1.844881, 1.845307, 1.848589, 1.861824, 1.887924, 1.923417, 1.992447, 2.017391,
                     2.038991, 2.071418, 2.129862, 2.166071, 2.747946, 1.768878, 1.946804, 2.157280, 2.245080,
                     2.087712, 1.820423, 1.907402, 2.200424, 2.181985, 2.221030]
    STSTEFAN_COUNTS = [0] * 26 + [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5]
    STSTEFAN_AREAS = [1.488923, 1.599765, 1.690057, 1.714709, 1.726957, 1.757419, 1.791569, 1.808079, 1.890679,
                      1.948752, 2.021423, 2.021726, 2.059094, 2.065921, 2.072363, 2.088126, 2.101806, 2.123316,
                      2.129358, 2.158377, 2.203595, 2.264995, 2.276287, 2.448357, 2.708561, 2.731780, 1.834966,
                      1.949574, 1.996594, 2.132053, 2.241751, 2.628096, 2.875178, 1.778265, 1.988550, 2.199333,
                      2.257071, 2.149337, 2.208404, 1.845604]

    def test_multimodal_cell_reaches_the_global_optimum(self):
        r = pop.estimate_zinb(self.LEMBACH_COUNTS, self.LEMBACH_AREAS)
        assert r.density_per_100ha == pytest.approx(57.313, rel=1e-3)      # glmmTMB dens_zinb
        assert r.se == pytest.approx(29.847, rel=1e-2)                     # glmmTMB se_zinb
        assert r.density_per_100ha > 57.0 and r.error is None              # not the 56.41 local optimum

    def test_identified_cell_matches_density_and_se(self):
        r = pop.estimate_zinb(self.STSTEFAN_COUNTS, self.STSTEFAN_AREAS)
        assert r.density_per_100ha == pytest.approx(32.672, rel=1e-3)
        assert r.se == pytest.approx(2.495, rel=1e-2)
        assert r.ci95[0] == pytest.approx(27.78, abs=0.05) and r.ci95[1] == pytest.approx(37.56, abs=0.05)
        assert r.error is None

    def test_naive_matches_r_exactly(self):
        assert pop.estimate_naive(self.LEMBACH_COUNTS, self.LEMBACH_AREAS).density_per_100ha == pytest.approx(39.481, rel=1e-4)


class TestEstimatePopulation:
    COUNTS = [3, 0, 5, 2, 0, 1, 4, 0]
    AREAS = [1.5, 2.0, 0.5, 1.0, 1.2, 0.8, 1.1, 1.7]

    def test_runs_the_requested_methods_only(self):
        r = pop.estimate_population(self.COUNTS, self.AREAS, methods=("naive",))
        assert r.naive is not None and r.bootstrap is None and r.zinb is None
        assert r.n_transects == 8 and r.n_zero_transects == 3 and r.total_count == 15

    def test_all_three_and_the_dict_form(self):
        r = pop.estimate_population(self.COUNTS, self.AREAS, n_boot=200)
        d = r.to_dict()
        assert set(d["estimates"]) == {"naive", "bootstrap", "zinb"}
        assert d["estimates"]["zinb"]["params"]["theta"] == r.zinb.theta

    def test_study_area_extrapolates(self):
        r = pop.estimate_population(self.COUNTS, self.AREAS, methods=("naive", "bootstrap"), study_area_ha=250.0)
        assert r.naive.abundance_study_area == pytest.approx(r.naive.density_per_100ha * 2.5)
        assert r.bootstrap.abundance_ci95[0] == pytest.approx(r.bootstrap.ci95[0] * 2.5)
        assert pop.estimate_population(self.COUNTS, self.AREAS, methods=("naive",)).naive.abundance_study_area is None
        with pytest.raises(ValueError):
            pop.estimate_population([1, 2], [1.0])


@requires_plugin
def test_matches_the_plugin_module():
    ppop = plugin_module("core.population")
    ref = ppop.estimate_zinb(TestZinbAgainstGlmmTMB.LEMBACH_COUNTS, TestZinbAgainstGlmmTMB.LEMBACH_AREAS)
    got = pop.estimate_zinb(TestZinbAgainstGlmmTMB.LEMBACH_COUNTS, TestZinbAgainstGlmmTMB.LEMBACH_AREAS)
    assert got.density_per_100ha == pytest.approx(ref["density_per_100ha"], rel=1e-9)
    assert got.se == pytest.approx(ref["se"], rel=1e-6)
    assert got.log_likelihood == pytest.approx(ref["log_likelihood"], rel=1e-9)
    b = pop.estimate_bootstrap(TestEstimatePopulation.COUNTS, TestEstimatePopulation.AREAS, 500, 3)
    rb = ppop.estimate_bootstrap(TestEstimatePopulation.COUNTS, TestEstimatePopulation.AREAS, 500, 3)
    assert b.density_per_100ha == rb["density_per_100ha"] and list(b.ci95) == rb["ci95"]
    tracks = [{"track_id": i, "last_frame": 0, "class_id": 0, "detection_center": [x, y, 0.0]}
              for i, (x, y) in enumerate([(50, 10), (50, 90), (50, 30), (20, 8)])]
    ra = ppop.assign_tracks(tracks, {1: [(0.0, 0.0), (100.0, 0.0)], 2: [(0.0, 100.0), (100.0, 100.0)]},
                            truncation=20.0, contains=ppop.shapely_area_predicate({1: _strips()[0], 2: _strips()[1]}))
    pts = [[50, 10], [50, 90], [50, 30], [20, 8]]
    a = pop.assign_to_transects(pts, LINES, truncation=20.0, inside=pop.points_in_geometries(pts, _strips()))
    assert [None if t < 0 else t + 1 for t in a.transect] == [r["transect_id"] for r in ra]
    assert a.outside.tolist() == [r["outside_fov"] for r in ra]
