# Changelog

## 1.0.1 - 2026-10-07

### Fixed
- **The render engine is the caller's choice again.** alfspy 3.0 merged the
  former `alfs_pytorch` fork back in, so one package now carries three engines
  (ModernGL, PyTorch, Vulkan) and two ray casters (Embree, Warp), selected with
  `$ALFS_ENGINE` / `$ALFS_RAYCASTER`. `bambi.util.render_context` still told the
  two old distributions apart by asking whether `make_torch_context` existed - a
  test that is now true on every machine, so it answered `torch` whatever was
  asked for and always built a torch context. `$ALFS_ENGINE` was ignored
  entirely, and on a machine that installed only `AlfsPy[moderngl]` rendering
  refused to start at all with "install AlfsPy[torch]". It now resolves through
  alfspy, and a failure names the engines that do work here.
- **Ray casting no longer rebuilds its acceleration structure per call.** alfspy
  3.0 builds one inside every `pixel_to_world_coord` handed a bare mesh, and
  every georeferencing call site casts the same DEM once per frame.
  `bambi.geo.georef` now passes a caster built once per mesh
  (`bambi.util.render_context.ray_caster_for`) - measured 22x on a 24-frame
  footprint loop, and `04_survey_analytics.ipynb` end to end from 52s to 12s.
- `trimesh` unpinned from `3.21.7` to `>=4.0,<6`, which alfspy 3.0 requires
  anyway. trimesh 3.x looks for the long-obsolete `pyembree` module and so
  silently falls back to a pure-Python ray intersector that returns hits in a
  **different index order** - a correctness difference under `bambi.geo.georef`,
  not merely a slow path.
- **`label_to_world_coordinates` uses the cached ray caster too.** It was the
  one helper outside `bambi.geo.georef` still handing alfspy the bare mesh, and
  it is what the QGIS plugin projects detections, tracks and FoV footprints
  through: about 0.25 s and 55 MB per call on a 265k-triangle DEM. A FoV run
  (one call per mask vertex per frame) looked hung, and a 5000-detection
  georeference climbed past 45 GB.
- **OpenCV undistortion no longer dies with a bare `Unknown exception`.**
  `bambi.geo.calibration.undistort_maps` / `remap` wrap
  `initUndistortRectifyMap` / `remap`: when OpenCV's parallel backend (the
  Concurrency Runtime on Windows) fails inside a host process such as QGIS, the
  call is retried single-threaded, and a remaining failure reports its inputs
  and the OpenCV build. The video and photo extractors use them.

### Changed
- `notebooks/_setup.py` installs `AlfsPy[<engine>,<raycaster>] @ v3.0.0` instead
  of the superseded `alfs_pytorch@v1.1.1`, and exports the pair it installed, so
  a notebook renders on the engine it was given. `ensure_environment` takes
  `engine=` / `raycaster=` with alfspy's own precedence - argument, then
  environment, then default - so `ALFS_ENGINE=vulkan` before the kernel starts
  is enough. The default stays `torch`, which needs no GL driver and is what the
  frozen results under `rendered/` were produced with.
- CI runs one unit leg per engine rather than one for `torch` plus one for
  ModernGL: with all three factories in a single package, a torch-only matrix
  passes whether or not the seam works.
- `bambi.util.render_context` gains `available_engines`, `raycast_backend`,
  `make_ray_caster` and `ray_caster_for`; `make_render_context` gains `engine=`.

### Verified
- All six notebooks execute on every engine and ray caster - 36 runs across the
  full 3x2 matrix, no failures.
- The suite passes on all three engines (578 tests each, including the QGIS
  plugin parity tier for georeferencing, GeoTIFF and tracking).
- Every alfspy symbol the package imports (33 of them) resolves against 3.0.0,
  and all 18 console entry points start.

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
