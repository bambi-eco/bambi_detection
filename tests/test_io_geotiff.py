# -*- coding: utf-8 -*-
"""bambi.io.geotiff - frame GeoTIFF + world file, average merge, overlap count."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from bambi.io import geotiff as io  # noqa: E402


def _img(v, h=20, w=30):
    return np.full((h, w, 3), v, dtype=np.uint8)


def test_frame_geotiff_round_trip_and_world_file(tmp_path):
    img = _img(120)
    valid = np.ones((20, 30), bool)
    valid[:5] = False
    p = io.write_frame_geotiff(tmp_path / "f.tiff", img, valid, (100.0, 200.0, 130.0, 220.0), 32633)
    back, bounds, epsg, nodata = io.read_geotiff(p)
    assert back.shape == (20, 30, 3) and bounds == pytest.approx((100, 200, 130, 220)) and epsg == 32633
    assert nodata == 0 and (back[:5] == 0).all() and (back[5:] == 120).all()
    tfw = (tmp_path / "f.tfw").read_text().splitlines()
    assert float(tfw[0]) == pytest.approx(1.0) and float(tfw[3]) == pytest.approx(-1.0)
    assert float(tfw[4]) == pytest.approx(100.5) and float(tfw[5]) == pytest.approx(219.5)
    grey = io.write_frame_geotiff(tmp_path / "g.tif", np.full((4, 4), 7, np.uint8), None, (0, 0, 4, 4), 32633,
                                  world_file=False)
    assert io.read_geotiff(grey)[0].shape == (4, 4) and not (tmp_path / "g.tfw").exists()


def test_merge_average_and_overlap_count(tmp_path):
    a = io.write_frame_geotiff(tmp_path / "a.tiff", _img(100), None, (0.0, 0.0, 30.0, 20.0), 32633)
    b = io.write_frame_geotiff(tmp_path / "b.tiff", _img(200), None, (15.0, 0.0, 45.0, 20.0), 32633)
    out = io.merge_average([a, b], tmp_path / "mosaic.tif")
    img, bounds, epsg, nodata = io.read_geotiff(out)
    assert bounds == pytest.approx((0, 0, 45, 20)) and img.shape == (20, 45, 3)
    assert (img[:, :15] == 100).all() and (img[:, 30:] == 200).all() and (img[:, 15:30] == 150).all()
    count, cb = io.count_overlaps([a, b])
    assert cb == pytest.approx((0, 0, 45, 20)) and count.dtype == np.uint16
    assert count[:, 0].max() == 1 and count[:, 20].max() == 2 and count.max() == 2


def test_write_geotiff_keeps_alpha_and_writes_sidecars(tmp_path):
    rgba = np.zeros((300, 400, 4), np.uint8)
    rgba[50:250, 100:300] = (10, 20, 30, 255)
    rgba[100:120, 150:170, 3] = 128                                    # partial coverage stays partial
    p = io.write_geotiff(tmp_path / "alfs.tif", rgba, (0.0, 0.0, 40.0, 30.0), 32633)
    img, bounds, epsg, nodata = io.read_geotiff(p)
    assert img.shape == (300, 400, 4) and nodata is None and epsg == 32633
    assert np.array_equal(img, rgba)                                    # alpha is band 4, untouched
    with rasterio.open(p) as src:
        assert src.descriptions == ("Red", "Green", "Blue", "Alpha")
        assert src.colorinterp[3] == rasterio.enums.ColorInterp.alpha
        assert src.overviews(1) == [2, 4, 8, 16] and src.profile["tiled"]
    assert (tmp_path / "alfs.tfw").exists() and "32633" in (tmp_path / "alfs.prj").read_text()
    small = io.write_geotiff(tmp_path / "s.tif", np.full((10, 10), 3, np.uint8), (0, 0, 1, 1), 32633, prj_file=False)
    with rasterio.open(small) as src:
        assert src.count == 1 and src.overviews(1) == []
    assert not (tmp_path / "s.prj").exists()
