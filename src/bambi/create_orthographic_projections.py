"""
create_orthographic_projections.py
------------------------------------
Creates orthographic projections for pre-exported thermal frames.

Inputs
------
- Frames in <frames_dir>/ named  <flight_id>_<frame_idx>.png
- matched_poses.json  – camera poses produced by PhotoPoseExtractor
- DEM .gltf + companion .json  – digital elevation model
- correction.json  – flight-specific pose corrections

Output
------
For every frame that has a matching pose entry a *_projected.png file is
written.  By default the projected files are placed alongside the originals
(same directory, _projected suffix).  Set OUTPUT_DIR to a different path to
separate them.

Usage
-----
Edit the USER CONFIGURATION section below, then run:
    python create_orthographic_projections.py
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

# Poses JSON written by PhotoPoseExtractor (matched_poses.json)
MATCHED_POSES_JSON = r"146_frames/matched_poses.json"

# DEM mesh file (GLTF) and its companion metadata JSON
PATH_TO_DEM = r"path/to/dem_mesh.gltf"          # e.g. dem_mesh_r2.gltf
PATH_TO_DEM_JSON = r"path/to/dem_mesh.json"      # set to "" to derive from gltf path

# Flight-specific correction JSON
PATH_TO_CORRECTION = r"path/to/correction.json"

# Where to save projected images.
# Set to "" (empty string) to write alongside originals in FRAMES_DIR.
OUTPUT_DIR = ""

# Skip frames whose projected output already exists
SKIP_EXISTING = True

# Limit the number of frames to project (set to -1 for all frames)
LIMIT = -1

# ── Orthographic camera settings ──────────────────────────────────────────
ORTHO_WIDTH  = 70    # metres – horizontal extent of the rendered area
ORTHO_HEIGHT = 70    # metres – vertical extent of the rendered area
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


def _load_correction(path: str) -> Transform:
    """Load the flight correction JSON and return an alfspy Transform."""
    with open(path, "r") as f:
        data = json.load(f)
    t = data.get("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    r = data.get("rotation",    {"x": 0.0, "y": 0.0, "z": 0.0})
    translation = Vector3([t["x"], t["y"], t["z"]], dtype="f4")
    eulers = Vector3([r["x"], r["y"], r["z"]], dtype="f4")
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

    # Fallback: derive mask size from the first available frame
    for entry in poses["images"]:
        candidate = os.path.join(frames_dir, os.path.basename(entry["imagefile"]))
        if os.path.exists(candidate):
            img = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                print(f"Using full-white fallback mask ({w}×{h}) derived from {candidate}")
                white = np.full((h, w), 255, dtype=np.uint8)
                return CtxShot._cvt_img(white)

    raise RuntimeError(
        "Could not determine mask dimensions – no frames found in FRAMES_DIR."
    )


def _resolve_image_path(image_metadata: dict, frames_dir: str) -> str | None:
    """
    Find the on-disk path for an image entry.

    Tries in order:
    1. frames_dir / imagefile  (as stored, possibly with subdirectory)
    2. frames_dir / basename(imagefile)  (plain filename fallback)
    """
    imagefile = image_metadata["imagefile"]
    for candidate in (
        os.path.join(frames_dir, imagefile),
        os.path.join(frames_dir, os.path.basename(imagefile)),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Render orthographic projections for extracted frames.')
    parser.add_argument("--frames-dir", dest="FRAMES_DIR", default=r"146_frames/thermal")
    parser.add_argument("--matched-poses-json", dest="MATCHED_POSES_JSON", default=r"146_frames/matched_poses.json")
    parser.add_argument("--path-to-dem", dest="PATH_TO_DEM", default=r"path/to/dem_mesh.gltf")
    parser.add_argument("--path-to-dem-json", dest="PATH_TO_DEM_JSON", default=r"path/to/dem_mesh.json")
    parser.add_argument("--path-to-correction", dest="PATH_TO_CORRECTION", default=r"path/to/correction.json")
    parser.add_argument("--output-dir", dest="OUTPUT_DIR", default="")
    parser.add_argument("--skip-existing", dest="SKIP_EXISTING", action="store_false", help="default: True")
    parser.add_argument("--limit", dest="LIMIT", type=int, default=-1)
    parser.add_argument("--ortho-width", dest="ORTHO_WIDTH", type=int, default=70)
    parser.add_argument("--ortho-height", dest="ORTHO_HEIGHT", type=int, default=70)
    parser.add_argument("--render-width", dest="RENDER_WIDTH", type=int, default=2048)
    parser.add_argument("--render-height", dest="RENDER_HEIGHT", type=int, default=2048)
    parser.add_argument("--fovy-fallback", dest="FOVY_FALLBACK", type=int, default=50)
    parser.add_argument("--aspect-ratio", dest="ASPECT_RATIO", type=int, default=1)
    args = parser.parse_args(argv)
    # The body below reads the module constants; apply the CLI overrides to
    # them so every former hard-coded value is settable from the command line.
    global FRAMES_DIR, MATCHED_POSES_JSON, PATH_TO_DEM, PATH_TO_DEM_JSON, PATH_TO_CORRECTION, OUTPUT_DIR, SKIP_EXISTING, LIMIT, ORTHO_WIDTH, ORTHO_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT, FOVY_FALLBACK, ASPECT_RATIO
    for _name, _value in vars(args).items():
        globals()[_name] = _value
    _run()


def _run() -> None:
    # ── Derive optional paths ────────────────────────────────────────────────
    path_to_dem      = PATH_TO_DEM
    path_to_dem_json = PATH_TO_DEM_JSON if PATH_TO_DEM_JSON else path_to_dem.replace(".gltf", ".json")
    output_dir       = OUTPUT_DIR if OUTPUT_DIR else FRAMES_DIR

    os.makedirs(output_dir, exist_ok=True)

    # ── Load poses ───────────────────────────────────────────────────────────
    with open(MATCHED_POSES_JSON, "r") as f:
        poses = json.load(f)

    images = poses["images"]
    frame_count = len(images)
    print(f"Loaded {frame_count} pose entries from {MATCHED_POSES_JSON}")

    # ── Load DEM metadata (origin offsets) ───────────────────────────────────
    with open(path_to_dem_json, "r") as f:
        dem_json = json.load(f)
    print(f"DEM origin: {dem_json.get('origin', 'n/a')}")

    # ── Load correction ──────────────────────────────────────────────────────
    correction = _load_correction(PATH_TO_CORRECTION)

    # ── Load mask ────────────────────────────────────────────────────────────
    mask_shot = _load_mask(poses, FRAMES_DIR)
    mask_height, mask_width = mask_shot.shape[:2]
    mask_texture = TextureData(mask_shot)
    render_resolution = Resolution(RENDER_WIDTH, RENDER_HEIGHT)

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
        resolution=render_resolution,
    )

    # ── Main rendering loop ──────────────────────────────────────────────────
    ctx = None
    mesh_data = None
    texture_data = None
    try:
        ctx = make_render_context()
        mesh_data, texture_data = read_gltf(path_to_dem)
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(mesh_data.vertices)

        processed = 0
        skipped   = 0
        missing   = 0

        total = frame_count if LIMIT < 0 else min(LIMIT, frame_count)

        for idx, image_metadata in enumerate(images):
            if 0 < LIMIT <= processed:
                print(f"Reached limit of {LIMIT} projections – stopping.")
                break

            image_path = _resolve_image_path(image_metadata, FRAMES_DIR)
            if image_path is None:
                imagefile = image_metadata["imagefile"]
                print(f"[{idx + 1}/{frame_count}] Missing: {imagefile} – skip")
                missing += 1
                continue

            stem      = Path(image_path).stem
            save_path = os.path.join(output_dir, stem + "_projected.png")

            if SKIP_EXISTING and os.path.exists(save_path):
                skipped += 1
                continue

            print(f"[{idx + 1}/{frame_count}] Projecting: {os.path.basename(image_path)}")

            shot     = None
            renderer = None
            try:
                # Normalise fovy to list format expected by create_shot
                meta = dict(image_metadata)
                meta["fovy"] = [_get_fovy(image_metadata)]

                shot = create_shot(image_path, meta, ctx, correction)
                single_shot_camera = make_camera(
                    mesh_aabb, [shot], settings,
                    rotation=Quaternion.from_matrix(
                        shot.get_view().inverse @ shot.get_correction().inverse
                    ),
                )
                renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)
                shot_loader = make_shot_loader([shot])
                renderer.project_shots(
                    shot_loader,
                    RenderResultMode.ShotOnly,
                    mask=mask_texture,
                    integral=False,
                    save=True,
                    release_shots=True,
                    save_name_iter=iter([save_path]),
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
