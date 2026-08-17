# -*- coding: utf-8 -*-
"""bambi.geo.dem - elevation grids become the pipeline's DEM meshes, index for index."""
import numpy as np
import pytest

from bambi.geo.dem import heightfield_grid_xy, heightfield_mesh, sample_heightfield


def reference_mesh(elevation, cell_width, cell_height, base):
    """The DEM tools' scalar loops (dem_upper_austria_downloader._create_gltf), verbatim."""
    height, width = elevation.shape
    elevation = np.nan_to_num(elevation, nan=base) - base
    vertices, indices = [], []
    for row in range(height):
        for col in range(width):
            vertices.append([col * cell_width, (height - 1 - row) * cell_height, float(elevation[row, col])])
    for row in range(height - 1):
        for col in range(width - 1):
            i00 = row * width + col; i10 = i00 + 1; i01 = (row + 1) * width + col; i11 = i01 + 1
            indices.extend([i00, i01, i10, i10, i01, i11])
    return np.array(vertices, np.float32), np.array(indices, np.uint32).reshape(-1, 3)


def test_matches_the_dem_tools_layout_exactly():
    rng = np.random.default_rng(3)
    z = rng.uniform(400, 460, size=(7, 9))
    z[2, 3] = np.nan
    v, f, base = heightfield_mesh(z, (2.5, 2.5))
    rv, rf = reference_mesh(z, 2.5, 2.5, np.nanmin(z))
    assert base == pytest.approx(np.nanmin(z))
    assert v.dtype == np.float32 and f.dtype == np.uint32
    assert np.array_equal(v, rv) and np.array_equal(f, rf)


def test_row_zero_is_the_northern_edge():
    xs, ys = heightfield_grid_xy((4, 3), (1.0, 2.0))
    assert np.allclose(xs, [0, 1, 2])
    assert np.allclose(ys, [6, 4, 2, 0])          # top raster row -> largest y


def test_negative_cell_height_is_taken_absolute():
    v1, _, _ = heightfield_mesh(np.zeros((3, 3)), (1.0, -1.0))
    v2, _, _ = heightfield_mesh(np.zeros((3, 3)), (1.0, 1.0))
    assert np.array_equal(v1, v2)


def test_explicit_base_elevation():
    v, _, base = heightfield_mesh(np.full((2, 2), 500.0), (1, 1), base_elevation=480.0)
    assert base == 480.0 and np.allclose(v[:, 2], 20.0)


def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        heightfield_mesh(np.zeros((1, 5)), (1, 1))
    with pytest.raises(ValueError):
        heightfield_mesh(np.full((3, 3), np.nan), (1, 1))


def test_sample_heightfield_lies_on_the_mesh_triangles_and_nan_outside():
    z = np.array([[0.0, 1.0], [2.0, 9.0]])          # row 0 = north (y=1), row 1 = south (y=0)
    got = sample_heightfield(z, (1, 1), [[0, 0], [1, 1], [0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [2, 0]],
                             base_elevation=0.0)
    assert got[0] == 2.0 and got[1] == 1.0
    assert got[2] == pytest.approx(1.5)             # on the diagonal: both triangles agree
    # lower-left triangle (i00, i01, i10) is the plane through z=0,2,1: no 9 in it
    assert got[3] == pytest.approx(0.0 + 0.25 * (1 - 0) + 0.25 * (2 - 0))
    # the other triangle (i10, i01, i11) contains the 9
    assert got[4] == pytest.approx(9 + 0.25 * (2 - 9) + 0.25 * (1 - 9))
    assert np.isnan(got[5])
    # and every sample really is on the trimesh surface
    trimesh = pytest.importorskip("trimesh")
    v, f, _ = heightfield_mesh(z, (1, 1), base_elevation=0.0)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    for (x, y), zz in zip([[0.25, 0.75], [0.75, 0.25], [0.5, 0.5]], got[2:5][[1, 2, 0]]):
        loc, _, _ = m.ray.intersects_location([[x, y, 100.0]], [[0, 0, -1.0]])
        assert loc[0][2] == pytest.approx(zz, abs=1e-5)
