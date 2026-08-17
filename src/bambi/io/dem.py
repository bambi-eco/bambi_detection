# -*- coding: utf-8 -*-
"""The file edge for DEMs: GLB meshes and their metadata JSON, GeoTIFF grids.

The pipeline stores a DEM as ``<name>.glb`` (the triangle mesh, DEM-local
metres, see :mod:`bambi.geo.dem` for the layout) next to ``<name>.json`` with
the raster's CRS/transform and the mesh origin, both projected and WGS84.
This module reads and writes those files and turns a GeoTIFF into the
elevation grid the mesh builder wants. No geometry lives here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bambi.geo.dem import heightfield_mesh
from bambi.geo.poses import Origin, make_origin

PathLike = Union[str, Path]

__all__ = ["DemMesh", "read_dem_mesh", "read_dem_metadata", "write_dem_mesh",
           "read_geotiff_elevation", "geotiff_to_dem", "dem_origin"]


@dataclass(frozen=True)
class DemMesh:
    """A DEM mesh plus what its metadata JSON knew about it."""
    mesh: Any                                       # trimesh.Trimesh, DEM-local metres
    origin: Optional[Origin]                        # None when no metadata was found
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def vertices(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.vertices, dtype=np.float64)

    @property
    def faces(self) -> NDArray[np.int64]:
        return np.asarray(self.mesh.faces, dtype=np.int64)


def _metadata_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".json")


def read_dem_metadata(path: PathLike) -> Dict[str, Any]:
    """The DEM's companion JSON as written by the DEM tools."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dem_origin(metadata: Dict[str, Any], epsg: Optional[int] = None) -> Origin:
    """The DEM-local frame's anchor from a metadata dict.

    Uses ``origin_wgs84`` (lat/lon/alt) and the raster ``crs`` (``EPSG:xxxxx``);
    ``epsg`` overrides the CRS when the metadata has none.
    """
    wgs = metadata.get("origin_wgs84") or {}
    if not {"latitude", "longitude"} <= set(wgs):
        raise ValueError("DEM metadata has no origin_wgs84 {latitude, longitude, altitude}")
    code = epsg
    if code is None:
        crs = str(metadata.get("crs", ""))
        if crs.upper().startswith("EPSG:"):
            code = int(crs.split(":", 1)[1])
    if code is None:
        raise ValueError("DEM metadata has no EPSG crs; pass epsg=")
    alt = wgs.get("altitude", metadata.get("origin", [0, 0, 0])[2])
    return make_origin(float(wgs["latitude"]), float(wgs["longitude"]), float(alt), int(code))


def _load_mesh(path: Path):
    """``trimesh.Trimesh`` from a GLB/GLTF.

    Prefers alfspy's own glTF parser - it is what the pipeline and the QGIS
    plugin feed to the ray-caster, so vertices come out bit-identical to
    theirs - and falls back to ``trimesh.load`` when alfspy is not installed.
    """
    import trimesh

    try:
        from alfspy.core.util.gltf import gltf_extract
    except ImportError:
        gltf_extract = None
    if gltf_extract is not None:
        mesh_data, _ = gltf_extract(str(path))
        if mesh_data is None:
            raise ValueError(f"No mesh in {path}")
        return trimesh.Trimesh(vertices=np.asarray(mesh_data.vertices), faces=np.asarray(mesh_data.indices),
                               process=False)
    return trimesh.load(str(path), force="mesh", process=False)


def read_dem_mesh(mesh_path: PathLike, metadata_path: Optional[PathLike] = None,
                  epsg: Optional[int] = None) -> DemMesh:
    """Load a DEM mesh (``.glb``/``.gltf``) and, when present, its metadata JSON.

    :param mesh_path: the mesh file
    :param metadata_path: default ``<mesh>.json``; missing metadata gives ``origin=None``
    :param epsg: CRS override when the metadata lacks one
    """
    mp = Path(mesh_path)
    mesh = _load_mesh(mp)
    meta_p = Path(metadata_path) if metadata_path is not None else _metadata_path(mp)
    metadata: Dict[str, Any] = read_dem_metadata(meta_p) if meta_p.exists() else {}
    origin = None
    if metadata:
        try:
            origin = dem_origin(metadata, epsg)
        except ValueError:
            origin = None
    return DemMesh(mesh=mesh, origin=origin, metadata=metadata, path=mp)


def _vertex_normals(vertices: NDArray[np.float32], faces: NDArray[np.uint32]) -> NDArray[np.float32]:
    v = vertices.astype(np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.where(n > 0, fn / np.where(n > 0, n, 1), 0)
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n > 0, n, 1)).astype(np.float32)


def write_dem_mesh(mesh_path: PathLike, vertices: ArrayLike, faces: ArrayLike,
                   metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Write a mesh as GLB (and its metadata JSON alongside, when given).

    Same file layout as the DEM tools (POSITION + NORMAL + uint32 indices in
    one binary buffer), written directly so it does not depend on the
    installed trimesh's exporter.
    """
    import struct

    mp = Path(mesh_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    v = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32).reshape(-1, 3))
    f = np.ascontiguousarray(np.asarray(faces, dtype=np.uint32).reshape(-1, 3))
    n = _vertex_normals(v, f)

    def pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
        return b + fill * ((4 - len(b) % 4) % 4)

    vb, nb, ib = pad4(v.tobytes()), pad4(n.tobytes()), pad4(f.tobytes())
    blob = vb + nb + ib
    gltf = {
        "asset": {"version": "2.0", "generator": "bambi-detection"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": int(len(v)), "type": "VEC3",
             "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": int(len(n)), "type": "VEC3",
             "min": n.min(axis=0).tolist(), "max": n.max(axis=0).tolist()},
            {"bufferView": 2, "componentType": 5125, "count": int(f.size), "type": "SCALAR",
             "min": [int(f.min())], "max": [int(f.max())]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": v.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": len(vb), "byteLength": n.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": len(vb) + len(nb), "byteLength": f.nbytes, "target": 34963},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    jb = pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    with open(mp, "wb") as fh:
        fh.write(b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jb) + 8 + len(blob)))
        fh.write(struct.pack("<I", len(jb)) + b"JSON" + jb)
        fh.write(struct.pack("<I", len(blob)) + b"BIN\x00" + blob)
    if metadata is not None:
        _metadata_path(mp).write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    return mp


def read_geotiff_elevation(path: PathLike, treat_zero_as_nodata: bool = True
                           ) -> Tuple[NDArray[np.float64], Tuple[float, float], Tuple[float, float], str, list]:
    """Band 1 of a GeoTIFF as a float grid with NaN for nodata.

    :return: ``elevation (H, W)``, ``cell_size (w, h)``, ``bottom_left (x, y)``
        in the raster CRS, the CRS string, and the 9-element affine transform
        the DEM tools store
    """
    import rasterio

    with rasterio.open(str(path)) as src:
        z = src.read(1).astype(np.float64)
        if src.nodata is not None:
            z = np.where(z == src.nodata, np.nan, z)
        if treat_zero_as_nodata:
            z = np.where(z == 0, np.nan, z)          # reprojection edge fill
        t = src.transform
        crs = str(src.crs)
        bounds = src.bounds
    return (z, (abs(float(t.a)), abs(float(t.e))), (float(bounds.left), float(bounds.bottom)), crs,
            [t.a, t.b, t.c, t.d, t.e, t.f, 0.0, 0.0, 1.0])


def geotiff_to_dem(geotiff_path: PathLike, mesh_path: PathLike, simplify: int = 1) -> DemMesh:
    """GeoTIFF -> ``.glb`` + ``.json`` in the pipeline's layout, and load it back.

    :param simplify: keep every n-th cell (the DEM tools' ``simplify_factor``)
    """
    from pyproj import Transformer

    z, (cw, ch), (ox, oy), crs, transform9 = read_geotiff_elevation(geotiff_path)
    rows, cols = z.shape
    s = max(1, int(simplify))
    zs = z[::s, ::s]
    vertices, faces, base = heightfield_mesh(zs, (cw * s, ch * s))
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(ox, oy)
    metadata = {
        "width": cols, "height": rows, "crs": crs, "transform": transform9,
        "origin": [ox, oy, base],
        "origin_wgs84": {"latitude": float(lat), "longitude": float(lon), "altitude": base},
    }
    write_dem_mesh(mesh_path, vertices, faces, metadata)
    return read_dem_mesh(mesh_path)
