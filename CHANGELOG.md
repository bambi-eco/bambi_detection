# Changelog

## 1.0.0 - 2026-08-18

The first release of bambi-detection as the *engine* under the BAMBI QGIS
plugin: every capability the plugin computes is now an importable, tested,
array-in / array-out function here, proven on the public BAMBI dataset by
executing notebooks and against the plugin's own output by parity tests.
Major version because the public surface is new; the console scripts and
the modules of 0.6.0 are unchanged.

### The engine contract
Public functions on `bambi.geo`, `bambi.tracking`, `bambi.survey`,
`bambi.render`, `bambi.testing` take arrays (plus scalars, frozen
dataclasses of arrays, alfspy/shapely objects) and return arrays where the
result is array-shaped. File formats live in `bambi.io.*` only. Enforced by
`tests/test_architecture.py`.

### Added
- `bambi.geo.poses` - DEM-local frames (`Origin`, `Poses`), geographic <->
  local, gimbal <-> pose rotation, grid convergence.
- `bambi.geo.calibration` - calibration/media resolution check, the
  extractor's undistortion recipe (`new_camera_matrix`,
  `fovy_after_undistortion`, `undistort_points/boxes`).
- `bambi.geo.camera` - poses -> alfspy cameras (`quaternion_from_drone_pose`),
  the installed alfspy's ray convention probed once, `world_to_pixel`.
- `bambi.geo.georef` - pixels/boxes/footprints onto the DEM, misses as NaN
  rows aligned with the input; `boxes_to_world_by_frame`.
- `bambi.geo.dem` - elevation grids -> the pipeline's mesh layout.
- `bambi.tracking.iou` - the built-in tracker (greedy / Hungarian / centre
  modes) and gap interpolation on `(N,)`/`(N,4|6)` arrays.
- `bambi.tracking.matching` - cross-modal thermal/RGB track matching
  (frame pairing by clock, bootstrapped affine, Hungarian assignment).
- `bambi.survey.*` - transects, perpendicular distances, KDE density and
  coverage grids, line-transect distance sampling, and the naive /
  bootstrap / zero-inflated negative binomial population estimators
  (reproduce the glmmTMB reference analysis).
- `bambi.render.*` - orthophotos, light-field integrals (also tiled over large
  extents with footprint-filtered shots), tiling, mask polygons and their
  ground footprints, the per-frame GeoTIFF recipe.
- `bambi.testing.synthetic` - terrain + markers + poses at any tilt for
  exact-truth tests.
- `bambi.io.*` - poses, calibration, corrections (four dialects), DEM
  (GLB + metadata, GeoTIFF), tracks tables, TRex tracklets, survey files,
  rasters, per-frame orthophoto GeoTIFFs (nodata rim, world file) and
  light-field GeoTIFFs (alpha band kept, overviews, `.prj`), orthomosaic
  merge; writers byte-identical to the plugin's formats.
- `bambi.util.render_context` - backend-neutral alfspy contexts (ModernGL or
  PyTorch build).
- 18 console commands (`bambi-pipeline`, `bambi-georeference-*`,
  `bambi-track`, `bambi-alfs`, ...) over importable `run()` functions.
- Notebooks `00`-`05` on public flights, executed in CI; a test suite
  (unit / slow / notebook tiers) and GitHub Actions CI on both backends.

### Changed
- Every module imports cleanly (no repo-root-relative imports, no
  `sys.exit` at import time); private path defaults became required
  arguments.
- Requires Python >= 3.9; either alfspy build (alfs_py >= 2.1.0 or
  alfs_pytorch >= 1.1.1); `rtree` added for the ray caster.

### Known differences to the QGIS plugin (as of this release)
- Cross-modal matching refits are judged on the same pair set and both
  starts are refined (the plugin can freeze on an exact three-point seed).
- Renders default to `quaternion_from_drone_pose` shots; the plugin still
  spells shots `'zyx'` (identical at nadir). `convention="zyx"` reproduces it.
- Coverage can be computed from footprints, not only from exported GeoTIFFs.

## 0.6.0 - 2026-08-13
- Works against either alfspy backend; pose rotations via
  `quaternion_from_drone_pose`.

## 0.5.0
- Last release before the backend-neutral change.
