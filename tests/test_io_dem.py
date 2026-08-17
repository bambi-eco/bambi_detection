# -*- coding: utf-8 -*-
"""bambi.io.dem - GLB + metadata JSON round trip, GeoTIFF -> DEM."""
import json

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("pyproj")

from bambi.geo.dem import heightfield_mesh  # noqa: E402
from bambi.io.dem import dem_origin, geotiff_to_dem, read_dem_mesh, write_dem_mesh  # noqa: E402

META = {
    "width": 5, "height": 4, "crs": "EPSG:32633",
    "transform": [2.5, 0, 500000.0, 0, -2.5, 5170000.0, 0, 0, 1],
    "origin": [500000.0, 5169990.0, 430.0],
    "origin_wgs84": {"latitude": 46.6667, "longitude": 15.0, "altitude": 430.0},
}


def test_write_then_read_round_trips_vertices_faces_and_origin(tmp_path):
    z = np.arange(20, dtype=float).reshape(4, 5) + 430.0
    v, f, base = heightfield_mesh(z, (2.5, 2.5))
    dem = read_dem_mesh(write_dem_mesh(tmp_path / "x_dem.glb", v, f, META))
    assert dem.vertices.shape == (20, 3) and dem.faces.shape == (24, 3)
    assert np.allclose(dem.vertices, v) and np.array_equal(dem.faces, f)
    assert dem.origin is not None and dem.origin.epsg == 32633
    assert dem.origin.latitude == 46.6667 and dem.origin.altitude == 430.0
    assert dem.metadata["crs"] == "EPSG:32633"
    assert (tmp_path / "x_dem.json").exists()


def test_missing_metadata_gives_no_origin(tmp_path):
    v, f, _ = heightfield_mesh(np.zeros((3, 3)), (1, 1))
    dem = read_dem_mesh(write_dem_mesh(tmp_path / "bare.glb", v, f))
    assert dem.origin is None and dem.metadata == {}


def test_dem_origin_needs_a_crs():
    meta = dict(META); meta.pop("crs")
    with pytest.raises(ValueError):
        dem_origin(meta)
    assert dem_origin(meta, epsg=32633).epsg == 32633
    with pytest.raises(ValueError):
        dem_origin({"crs": "EPSG:32633"})


def test_geotiff_to_dem_reproduces_the_dem_tools_metadata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    z = (np.arange(30, dtype=np.float32).reshape(5, 6) + 400.0)
    z[0, 0] = 0.0                                          # edge fill -> nodata
    tif = tmp_path / "dem.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=5, width=6, count=1, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(500000.0, 5170000.0, 2.0, 2.0)) as dst:
        dst.write(z, 1)

    dem = geotiff_to_dem(tif, tmp_path / "out.glb")
    meta = json.loads((tmp_path / "out.json").read_text())
    assert meta["width"] == 6 and meta["height"] == 5 and meta["crs"] == "EPSG:32633"
    assert meta["transform"][:6] == [2.0, 0.0, 500000.0, 0.0, -2.0, 5170000.0]
    assert meta["origin"][:2] == [500000.0, 5170000.0 - 10.0]        # bottom-left
    assert meta["origin"][2] == pytest.approx(401.0)                  # min valid (the 0 was nodata)
    assert dem.vertices.shape == (30, 3)
    # the nodata cell is filled with the base (401) -> z 0; so is the 401 cell; then 402 - 401
    assert dem.vertices[0, 2] == 0.0 and dem.vertices[1, 2] == 0.0 and dem.vertices[2, 2] == pytest.approx(1.0)
    # origin_wgs84 is where the bottom-left corner really is
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True).transform(500000.0, 5169990.0)
    assert dem.origin.latitude == pytest.approx(lat) and dem.origin.longitude == pytest.approx(lon)
