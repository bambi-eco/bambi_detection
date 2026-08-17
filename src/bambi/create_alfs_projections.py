"""
create_alfs_projections.py
--------------------------
Creates Airborne Light Field Sampling (ALFS) projections for pre-exported
thermal frames.

For every central frame the script assembles a light field from N neighboring
frames (before + after it in matched_poses.json), then renders them jointly via
renderer.render_integral to produce a single de-blurred orthographic output.

Inputs
------
- Frames in <FRAMES_DIR>/ named  <flight_id>_<frame_idx>.png
- matched_poses.json  – camera poses produced by PhotoPoseExtractor
                        (entries must be sorted by acquisition time)
- DEM .gltf + companion .json  – digital elevation model
- correction.json  – flight-specific pose corrections

Output
------
For every eligible central frame a *_alfs.png file is written to OUTPUT_DIR
(defaults to FRAMES_DIR when left empty).

Usage
-----
Edit the USER CONFIGURATION section below, then run:
    python create_alfs_projections.py
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from alfspy.core.geo import Transform
from alfspy.core.rendering import Resolution, Renderer, RenderResultMode, TextureData
from alfspy.core.rendering import CtxShot
from alfspy.core.util.geo import get_aabb
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import (
    make_camera, make_shot_loader,
    process_render_data, read_gltf, release_all,
)
from bambi.util.render_context import make_render_context
from pyrr import Quaternion, Vector3
from trimesh import Trimesh

from bambi.util.projection_util import create_shot

# ─────────────────────────── USER CONFIGURATION ─────────────────────────────

# Folder that contains the exported thermal frames
FRAMES_DIR = r"146_frames/thermal"

# Poses JSON written by PhotoPoseExtractor (entries sorted by time)
MATCHED_POSES_JSON = r"146_frames/matched_poses.json"

# DEM mesh file (GLTF) and its companion metadata JSON
PATH_TO_DEM      = r"path/to/dem_mesh.gltf"   # e.g. dem_mesh_r2.gltf
PATH_TO_DEM_JSON = r""                          # leave "" to auto-derive from gltf path

# Flight-specific correction JSON
PATH_TO_CORRECTION = r"path/to/correction.json"

# Where to save projected images.
# Set to "" (empty string) to write alongside originals in FRAMES_DIR.
OUTPUT_DIR = ""

# Skip frames whose ALFS output already exists
SKIP_EXISTING = True

# Limit the number of central frames to project (-1 = all)
LIMIT = -1

# ── Light-field (ALFS) settings ───────────────────────────────────────────
# Number of neighboring frames to include on each side of the central frame.
# The central frame is never counted here, so the total light field size is
#   (NUM_NEIGHBORS // NEIGHBOR_SAMPLE_RATE) * 2  +  1  frames.
NUM_NEIGHBORS = 100

# Use every Nth neighbor (1 = all, 2 = every other, 10 = every 10th, …).
NEIGHBOR_SAMPLE_RATE = 10

# ── Orthographic camera settings ──────────────────────────────────────────
ORTHO_WIDTH   = 70    # metres – horizontal extent of the rendered area
ORTHO_HEIGHT  = 70    # metres – vertical extent of the rendered area
RENDER_WIDTH  = 2048  # output image width  (pixels)
RENDER_HEIGHT = 2048  # output image height (pixels)
FOVY_FALLBACK = 50    # degrees – used when fovy is absent from poses
ASPECT_RATIO  = 1

# ────────────────────────────────────────────────────────────────────────────


def _get_fovy(image_metadata: dict, fallback: float = FOVY_FALLBACK) -> float:
    """Return fovy as a scalar, handling both tuple/list and plain-float storage."""
    fov = image_metadata.get("fovy", fallback)
    if isinstance(fov, (list, tuple)):
        return float(fov[0])
    return float(fov)


def _patch_meta(image_metadata: dict) -> dict:
    """Return a copy of image_metadata with fovy normalised to a list."""
    meta = dict(image_metadata)
    meta["fovy"] = [_get_fovy(image_metadata)]
    return meta


def _load_correction(path: str) -> Transform:
    with open(path, "r") as f:
        data = json.load(f)
    t = data.get("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    r = data.get("rotation",    {"x": 0.0, "y": 0.0, "z": 0.0})
    translation = Vector3([t["x"], t["y"], t["z"]], dtype="f4")
    eulers      = Vector3([r["x"], r["y"], r["z"]], dtype="f4")
    return Transform(translation, Quaternion.from_eulers(eulers))


def _load_mask(poses: dict, frames_dir: str) -> np.ndarray:
    """
    Load the undistortion mask.

    Priority:
    1. Path stored in poses['mask']  (written by PhotoPoseExtractor)
    2. Full-white mask derived from the first available frame's dimensions
    """
    mask_file = poses.get("mask")
    if mask_file:
        mask_path = os.path.join(frames_dir, mask_file)
        if os.path.exists(mask_path):
            img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"Using mask: {mask_path}")
                return CtxShot._cvt_img(img)
        print(f"Warning: mask '{mask_file}' listed in poses but not found – using full-white fallback.")

    for entry in poses["images"]:
        candidate = os.path.join(frames_dir, os.path.basename(entry["imagefile"]))
        if os.path.exists(candidate):
            img = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                print(f"Using full-white fallback mask ({w}×{h}) derived from {candidate}")
                white = np.full((h, w), 255, dtype=np.uint8)
                return CtxShot._cvt_img(white)

    raise RuntimeError("Could not determine mask dimensions – no frames found in FRAMES_DIR.")


def _resolve_image_path(image_metadata: dict, frames_dir: str) -> str | None:
    """Locate the on-disk path for an image entry (exact path, then basename fallback)."""
    imagefile = image_metadata["imagefile"]
    for candidate in (
        os.path.join(frames_dir, imagefile),
        os.path.join(frames_dir, os.path.basename(imagefile)),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _collect_neighbor_shots(
    images: list,
    central_idx: int,
    frames_dir: str,
    ctx,
    correction: Transform,
) -> tuple[list, list]:
    """
    Build shot lists for the frames before and after *central_idx*.

    Returns (shots_before, shots_after).  Missing frames are silently skipped.
    Fixes the bug present in bambi_detection.py where the central frame's
    path and metadata were used for all neighbor shots instead of each
    neighbor's own data.
    """
    shots_before = []
    shots_after  = []

    # ── frames before central ──────────────────────────────────────────────
    for step in range(0, NUM_NEIGHBORS):
        if step % NEIGHBOR_SAMPLE_RATE != 0:
            continue
        nb_idx = central_idx - NUM_NEIGHBORS + step
        if nb_idx < 0:
            continue
        nb_meta = images[nb_idx]
        nb_path = _resolve_image_path(nb_meta, frames_dir)
        if nb_path is None:
            continue
        shots_before.append(create_shot(nb_path, _patch_meta(nb_meta), ctx, correction))

    # ── frames after central ───────────────────────────────────────────────
    for step in range(1, NUM_NEIGHBORS + 1):
        if step % NEIGHBOR_SAMPLE_RATE != 0:
            continue
        nb_idx = central_idx + step
        if nb_idx >= len(images):
            continue
        nb_meta = images[nb_idx]
        nb_path = _resolve_image_path(nb_meta, frames_dir)
        if nb_path is None:
            continue
        shots_after.append(create_shot(nb_path, _patch_meta(nb_meta), ctx, correction))

    return shots_before, shots_after


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Render ALFS light-field integrals for extracted frames.')
    parser.add_argument("--frames-dir", dest="FRAMES_DIR", default=r"146_frames/thermal")
    parser.add_argument("--matched-poses-json", dest="MATCHED_POSES_JSON", default=r"146_frames/matched_poses.json")
    parser.add_argument("--path-to-dem", dest="PATH_TO_DEM", default=r"path/to/dem_mesh.gltf")
    parser.add_argument("--path-to-dem-json", dest="PATH_TO_DEM_JSON", default=r"")
    parser.add_argument("--path-to-correction", dest="PATH_TO_CORRECTION", default=r"path/to/correction.json")
    parser.add_argument("--output-dir", dest="OUTPUT_DIR", default="")
    parser.add_argument("--skip-existing", dest="SKIP_EXISTING", action="store_false", help="default: True")
    parser.add_argument("--limit", dest="LIMIT", type=int, default=-1)
    parser.add_argument("--num-neighbors", dest="NUM_NEIGHBORS", type=int, default=100)
    parser.add_argument("--neighbor-sample-rate", dest="NEIGHBOR_SAMPLE_RATE", type=int, default=10)
    parser.add_argument("--ortho-width", dest="ORTHO_WIDTH", type=int, default=70)
    parser.add_argument("--ortho-height", dest="ORTHO_HEIGHT", type=int, default=70)
    parser.add_argument("--render-width", dest="RENDER_WIDTH", type=int, default=2048)
    parser.add_argument("--render-height", dest="RENDER_HEIGHT", type=int, default=2048)
    parser.add_argument("--fovy-fallback", dest="FOVY_FALLBACK", type=int, default=50)
    parser.add_argument("--aspect-ratio", dest="ASPECT_RATIO", type=int, default=1)
    args = parser.parse_args(argv)
    # The body below reads the module constants; apply the CLI overrides to
    # them so every former hard-coded value is settable from the command line.
    global FRAMES_DIR, MATCHED_POSES_JSON, PATH_TO_DEM, PATH_TO_DEM_JSON, PATH_TO_CORRECTION, OUTPUT_DIR, SKIP_EXISTING, LIMIT, NUM_NEIGHBORS, NEIGHBOR_SAMPLE_RATE, ORTHO_WIDTH, ORTHO_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT, FOVY_FALLBACK, ASPECT_RATIO
    for _name, _value in vars(args).items():
        globals()[_name] = _value
    _run()


def _run() -> None:
    path_to_dem      = PATH_TO_DEM
    path_to_dem_json = PATH_TO_DEM_JSON if PATH_TO_DEM_JSON else path_to_dem.replace(".gltf", ".json")
    output_dir       = OUTPUT_DIR if OUTPUT_DIR else FRAMES_DIR

    os.makedirs(output_dir, exist_ok=True)

    # ── Load poses ───────────────────────────────────────────────────────────
    with open(MATCHED_POSES_JSON, "r") as f:
        poses = json.load(f)

    images      = poses["images"]
    frame_count = len(images)
    print(f"Loaded {frame_count} pose entries from {MATCHED_POSES_JSON}")

    # Central frames start at NUM_NEIGHBORS so that every central frame has a
    # full preceding neighborhood.  Frames before this offset are used only as
    # neighbors and do not produce their own ALFS output.
    start_idx   = NUM_NEIGHBORS
    eligible    = frame_count - start_idx
    if eligible <= 0:
        raise RuntimeError(
            f"Not enough frames ({frame_count}) for NUM_NEIGHBORS={NUM_NEIGHBORS}. "
            "Reduce NUM_NEIGHBORS or add more poses."
        )
    print(f"Eligible central frames: {eligible}  (indices {start_idx}–{frame_count - 1})")
    print(f"Light-field size per frame: up to {(NUM_NEIGHBORS // NEIGHBOR_SAMPLE_RATE) * 2 + 1} shots")

    # ── Load DEM metadata ────────────────────────────────────────────────────
    with open(path_to_dem_json, "r") as f:
        dem_json = json.load(f)
    print(f"DEM origin: {dem_json.get('origin', 'n/a')}")

    # ── Load correction ──────────────────────────────────────────────────────
    correction = _load_correction(PATH_TO_CORRECTION)

    # ── Load mask ────────────────────────────────────────────────────────────
    mask_shot    = _load_mask(poses, FRAMES_DIR)
    mask_texture = TextureData(mask_shot)
    render_res   = Resolution(RENDER_WIDTH, RENDER_HEIGHT)

    # ── Rendering settings ───────────────────────────────────────────────────
    settings = BaseSettings(
        count=frame_count,
        initial_skip=0,
        add_background=False,
        camera_position_mode=CameraPositioningMode.FirstShot,
        fovy=FOVY_FALLBACK,
        aspect_ratio=ASPECT_RATIO,
        orthogonal=True,
        ortho_size=(ORTHO_WIDTH, ORTHO_HEIGHT),
        correction=correction,
        resolution=render_res,
    )

    # ── Main rendering loop ──────────────────────────────────────────────────
    ctx         = None
    mesh_data   = None
    texture_data = None
    try:
        ctx = make_render_context()
        mesh_data, texture_data = read_gltf(path_to_dem)
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(mesh_data.vertices)

        processed = 0
        skipped   = 0
        missing   = 0

        for central_idx in range(start_idx, frame_count):
            if 0 < LIMIT <= processed:
                print(f"Reached limit of {LIMIT} projections – stopping.")
                break

            image_metadata = images[central_idx]
            image_path     = _resolve_image_path(image_metadata, FRAMES_DIR)

            if image_path is None:
                print(f"[{central_idx + 1}/{frame_count}] Missing: {image_metadata['imagefile']} – skip")
                missing += 1
                continue

            stem      = Path(image_path).stem
            save_path = os.path.join(output_dir, stem + "_alfs.png")

            if SKIP_EXISTING and os.path.exists(save_path):
                skipped += 1
                continue

            print(f"[{central_idx + 1}/{frame_count}] ALFS: {os.path.basename(image_path)}")

            shot     = None
            renderer = None
            try:
                # ── Central shot ───────────────────────────────────────────
                meta = _patch_meta(image_metadata)
                shot = create_shot(image_path, meta, ctx, correction)

                single_shot_camera = make_camera(
                    mesh_aabb, [shot], settings,
                    rotation=Quaternion.from_matrix(
                        shot.get_view().inverse @ shot.get_correction().inverse
                    ),
                )
                renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)

                # ── Neighbor shots ─────────────────────────────────────────
                shots_before, shots_after = _collect_neighbor_shots(
                    images, central_idx, FRAMES_DIR, ctx, correction
                )
                print(
                    f"    neighbors: {len(shots_before)} before + "
                    f"1 central + {len(shots_after)} after"
                )

                all_shots  = shots_before + [shot] + shots_after
                shot_loader = make_shot_loader(all_shots)

                renderer.render_integral(
                    shot_loader,
                    mask=mask_texture,
                    save=True,
                    release_shots=True,
                    save_name=save_path,
                )
                print(f"    -> {save_path}")
                processed += 1

            finally:
                release_all(renderer)

    finally:
        release_all(ctx)
        if mesh_data is not None:
            del mesh_data
        if texture_data is not None:
            del texture_data

    print(
        f"\nFinished. Projected: {processed} | Skipped (existing): {skipped} | Missing: {missing}"
    )


if __name__ == "__main__":
    main()
