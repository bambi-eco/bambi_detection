# -*- coding: utf-8 -*-
"""bambi.survey.density - KDE and coverage grids, against the plugin's density body."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from bambi.survey.density import NODATA, coverage_grid, grid_extent, kde_grid  # noqa: E402
from tests.plugin_oracle import plugin_processor, requires_plugin  # noqa: E402


def test_kde_conserves_points_and_reports_per_hectare():
    pts = np.array([[100, 100], [130, 90], [500, 480]], float)
    g = kde_grid(pts, cell_size=5.0, bandwidth=20.0, nodata=None)
    # sum(density) * cell_area / 10000 == number of points (kernel mass conserved inside the padding)
    assert g.values.astype(float).sum() * g.cell_size ** 2 / 10000.0 == pytest.approx(3.0, rel=5e-3)   # 3 sigma pad
    assert g.bounds == (100 - 60, 90 - 60, 500 + 60, 480 + 60)
    assert g.values.dtype == np.float32
    with_nodata = kde_grid(pts, 5.0, 20.0)
    assert (with_nodata.values == NODATA).any() and with_nodata.values.max() == pytest.approx(g.values.max())
    r, c = with_nodata.cell_of([[100, 100]])
    assert with_nodata.values[r[0], c[0]] > 0


def test_kde_arguments():
    with pytest.raises(ValueError):
        kde_grid(np.zeros((0, 2)), 5, 5)
    with pytest.raises(ValueError):
        kde_grid([[0, 0]], 0, 5)
    with pytest.raises(ValueError):
        kde_grid([[0, 0]], 5, 0)


def test_grid_extent_caps_the_size():
    bounds, cs, w, h = grid_extent([[0, 0], [100000, 100]], cell_size=1.0, max_dim=1000)
    assert w == 1000 and cs == pytest.approx(100.0)


def test_coverage_counts_overlaps():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[5, 5], [15, 5], [15, 15], [5, 15]]
    g = coverage_grid([a, b], cell_size=1.0)
    assert g.bounds == (0, 0, 15, 15) and g.values.shape == (15, 15) and g.values.dtype == np.uint16
    assert g.values.max() == 2 and int((g.values == 2).sum()) == 25 and int((g.values >= 1).sum()) == 175
    r, c = g.cell_of([[7.5, 7.5], [1.0, 1.0], [14, 14]])
    assert g.values[r, c].tolist() == [2, 1, 1]
    # NaN corners are dropped, degenerate footprints skipped
    g2 = coverage_grid([a, [[np.nan, np.nan], [1, 1]]], cell_size=1.0)
    assert g2.values.max() == 1
    with pytest.raises(ValueError):
        coverage_grid([[[0, 0], [1, 1]]], 1.0)


@requires_plugin
def test_kde_matches_the_plugin_density_step(tmp_path):
    """Run the plugin's _run_density_heatmap_once with its point loader patched,
    read the raster it writes, compare cell for cell."""
    rasterio = pytest.importorskip("rasterio")
    proc = plugin_processor()
    rng = np.random.default_rng(9)
    pts = np.column_stack([rng.uniform(1000, 1400, 80), rng.uniform(5000, 5300, 80)])
    proc._collect_analytics_points = lambda config, source, log_fn=None: ([tuple(p) for p in pts], "t")
    out = proc._run_density_heatmap_once({"target_folder": str(tmp_path), "density_source": "detections",
                                          "density_cell_size": 4.0, "density_bandwidth": 15.0,
                                          "target_epsg": 32633})
    with rasterio.open(out) as src:
        ref = src.read(1)
        b = src.bounds
    g = kde_grid(pts, cell_size=4.0, bandwidth=15.0)
    assert g.values.shape == ref.shape
    assert np.allclose(g.bounds, (b.left, b.bottom, b.right, b.top))
    assert np.array_equal(g.values, ref)
