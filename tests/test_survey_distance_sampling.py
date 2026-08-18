# -*- coding: utf-8 -*-
"""bambi.survey.distance_sampling - MLE detection functions, against theory and the plugin."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from bambi.survey import distance_sampling as ds  # noqa: E402
from tests.plugin_oracle import plugin_processor, requires_plugin  # noqa: E402


def _half_normal_sample(sigma, n, w, seed=1):
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(0, sigma, size=n * 3))
    return x[x <= w][:n]


def test_half_normal_recovers_sigma_and_esw():
    x = _half_normal_sample(12.0, 2000, 60.0)
    fit = ds.fit_detection_function(x, 60.0, "half-normal")
    assert fit.name == "half-normal" and fit.n_params == 1
    assert fit.params[0] == pytest.approx(12.0, rel=0.06)
    # ESW of an (almost untruncated) half-normal is sigma * sqrt(pi/2)
    assert fit.esw == pytest.approx(12.0 * math.sqrt(math.pi / 2), rel=0.06)
    assert 0 < fit.cv_esw < 0.05
    assert fit.g(0.0) == pytest.approx(1.0) and 0 < fit.g(20.0) < 1


def test_hazard_rate_fits_and_aic_prefers_the_right_shape():
    x = _half_normal_sample(12.0, 1500, 60.0, seed=3)
    hn = ds.fit_detection_function(x, 60.0, "half-normal")
    hr = ds.fit_detection_function(x, 60.0, "hazard-rate")
    assert hr.n_params == 2 and hr.params[1] > 1.0
    assert hn.aic <= hr.aic + 2.5                    # one extra parameter buys at most ~2 AIC on HN data
    with pytest.raises(ValueError):
        ds.fit_detection_function(x, 60.0, "uniform")
    assert ds.fit_detection_function([1.0], 60.0) is None


def test_lognormal_ci_and_truncation():
    lo, hi = ds.lognormal_ci(10.0, 0.2)
    assert lo < 10 < hi and lo * hi == pytest.approx(100.0)
    assert ds.lognormal_ci(0.0, 0.2) == (0.0, 0.0) and ds.lognormal_ci(5.0, 0.0) == (5.0, 5.0)
    assert ds.truncation_distance([1, 2, 3], 7.5) == 7.5
    assert ds.truncation_distance(np.arange(101.0), None) == pytest.approx(95.0)


def test_estimate_density_bookkeeping():
    x = _half_normal_sample(10.0, 400, 100.0)
    r = ds.estimate_density(np.concatenate([x, [np.nan, -1.0]]), transect_length=5000.0)
    assert r.n_before_truncation == 400 and r.n <= 400
    assert r.truncation_m == pytest.approx(np.percentile(x, 95))
    assert r.detection_probability == pytest.approx(r.effective_strip_width_m / r.truncation_m)
    assert r.density_per_km2 == pytest.approx(r.n / (2 * r.effective_strip_width_m * 5000.0) * 1e6)
    assert r.abundance_in_covered_area == pytest.approx(r.n / r.detection_probability)
    assert r.density_ci95[0] < r.density_per_km2 < r.density_ci95[1]
    assert r.curve_x.shape == (60,) and r.curve_g[0] == pytest.approx(1.0)
    assert r.histogram_counts.sum() == r.n
    assert r.best.name in ("half-normal", "hazard-rate") and len(r.models) == 2
    with pytest.raises(ValueError):
        ds.estimate_density([1.0], 100.0)
    with pytest.raises(ValueError):
        ds.estimate_density(x, 0.0)


@requires_plugin
def test_matches_the_plugin_fits():
    proc = plugin_processor()
    x = _half_normal_sample(15.0, 300, 70.0, seed=7)
    for name in ("half-normal", "hazard-rate"):
        ref = proc._fit_detection_function(name, x, 70.0)
        got = ds.fit_detection_function(x, 70.0, name)
        assert got.esw == pytest.approx(ref["esw"], rel=1e-9)
        assert got.aic == pytest.approx(ref["aic"], rel=1e-9)
        assert got.cv_esw == pytest.approx(ref["cv_esw"], rel=1e-6)
        assert got.params[0] == pytest.approx(ref["params"]["sigma"], rel=1e-9)
        if name == "hazard-rate":
            assert got.params[1] == pytest.approx(ref["params"]["b"], rel=1e-9)
        assert np.allclose(got.g(np.linspace(0, 70, 11)), ref["g"](np.linspace(0, 70, 11)))
    assert ds.lognormal_ci(12.3, 0.31) == pytest.approx(tuple(proc._lognormal_ci(12.3, 0.31)))
