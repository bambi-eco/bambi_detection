#!/usr/bin/env python3
"""
Georeference SAM3 Polygons Script

This script reads local polygon files (created by the SAM3 segmentation script),
georeferences them using camera poses and a DEM mesh, and writes the results
to new polygon files with global (world) coordinates.

Input format (per line):
    <object_type> <num_points> <x1> <y1> <x2> <y2> ... <xN> <yN>

Output format (per line):
    <object_type> <num_points> <X1> <Y1> <Z1> <X2> <Y2> <Z2> ... <XN> <YN> <ZN>

Usage:
    python georeference_polygons.py --source ./local_polygons --target ./georeferenced_polygons \
        --correction-folder ./correction_data --flight-id 223

Requirements:
    - alfspy library (for rendering and projection)
    - pyproj (for CRS transformations)
    - trimesh (for mesh handling)
    - numpy, scipy
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

try:
    from alfspy.core.rendering import Resolution, Camera
    from alfspy.core.util.geo import get_aabb
    from alfspy.render.render import make_mgl_context, read_gltf, process_render_data, release_all
    from pyrr import Quaternion, Vector3
    from trimesh import Trimesh
    from pyproj import CRS, Transformer
    from pyproj.enums import TransformDirection
    from src.bambi.util.projection_util import label_to_world_coordinates
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please ensure alfspy, pyproj, trimesh, and pyrr are installed.")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default input resolution (can be overridden via command line)
DEFAULT_INPUT_RESOLUTION = Resolution(1024, 1024)

# Smoothing parameters for camera poses
DEFAULT_WINDOW_LENGTH = 11
DEFAULT_POLYORDER = 2


# =============================================================================
# POLYGON FILE I/O
# =============================================================================

def parse_polygon_line(line: str) -> Optional[Tuple[str, List[Tuple[float, float]]]]:
    """
    Parse a single line from a SAM3 polygon file.

    Format: <object_type> <num_points> <x1> <y1> <x2> <y2> ... <xN> <yN>
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) < 3:
        return None

    object_type = parts[0]
    try:
        num_points = int(parts[1])
    except ValueError:
        return None

    # Expected number of coordinate values: num_points * 2 (x and y for each point)
    expected_coords = num_points * 2
    coord_parts = parts[2:]

    if len(coord_parts) < expected_coords:
        return None

    try:
        coords = [float(c) for c in coord_parts[:expected_coords]]
        polygon = [(coords[i], coords[i + 1]) for i in range(0, expected_coords, 2)]
        return (object_type, polygon)
    except ValueError:
        return None


def read_polygon_file(file_path: Path) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Read all polygons from a SAM3 polygon file."""
    polygons = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            result = parse_polygon_line(line)
            if result is not None:
                polygons.append(result)
    return polygons


def format_georeferenced_polygon_line(
        object_type: str,
        georeferenced_points: List[Optional[Tuple[float, float, float]]]
) -> str:
    """
    Format a georeferenced polygon as a line for the output file.

    Format: <object_type> <num_points> <X1> <Y1> <Z1> <X2> <Y2> <Z2> ... <XN> <YN> <ZN>
    """
    # Filter out None values (points that couldn't be georeferenced)
    valid_points = [p for p in georeferenced_points if p is not None]

    if len(valid_points) == 0:
        return f"{object_type} 0"

    num_points = len(valid_points)
    coords = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in valid_points)
    return f"{object_type} {num_points} {coords}"


def write_georeferenced_polygon_file(
        output_path: Path,
        georeferenced_polygons: List[Tuple[str, List[Optional[Tuple[float, float, float]]]]],
        header_comment: str = None
) -> int:
    """Write georeferenced polygons to an output file."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
            f.write("# Format: object_type num_points X1 Y1 Z1 X2 Y2 Z2 ... XN YN ZN\n")

        for object_type, points in georeferenced_polygons:
            line = format_georeferenced_polygon_line(object_type, points)
            f.write(line + "\n")
            count += 1

    return count


def extract_frame_index(filename: str) -> int:
    """
    Extract the frame index from a filename.
    Assumes the last sequence of digits in the filename is the frame index.
    Example: 'segmentation_0123.txt' -> 123
    """
    # Find all digit sequences
    matches = re.findall(r'\d+', filename)
    if not matches:
        raise ValueError(f"Could not extract frame index from filename: {filename}")

    # Return the last one found (usually the safest bet for frame numbers)
    return int(matches[-1])


# =============================================================================
# CAMERA AND POSE HANDLING
# =============================================================================

def smooth_pose_positions_savgol(poses: Dict, window_length: int = 11, polyorder: int = 2):
    """Smooth the drone GPS positions in-place using a Savitzky–Golay filter."""
    images = poses.get("images", [])
    n = len(images)
    if n == 0:
        return

    # Ensure valid parameters
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_length < 3:
        return
    if window_length % 2 == 0:
        window_length += 1  # must be odd

    # Extract Nx3 matrix of positions
    positions = np.array([img["location"] for img in images], dtype=float)

    # Apply SavGol smoothing along time axis
    smoothed = savgol_filter(
        positions,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode="interp"
    )

    # Write back to poses in-place
    for img, loc in zip(images, smoothed):
        img["location"] = loc.tolist()


def get_camera_for_frame(
        matched_poses: Dict,
        frame_idx: int,
        cor_rotation_eulers: Vector3,
        cor_translation: Vector3,
        overrule_fov: Optional[float] = None
) -> Camera:
    """Create a Camera object for a specific frame."""
    # Ensure frame index is within bounds
    if frame_idx < 0 or frame_idx >= len(matched_poses['images']):
        raise IndexError(f"Frame index {frame_idx} is out of bounds for pose data.")

    cur_frame_data = matched_poses['images'][frame_idx]
    fovy = cur_frame_data['fovy'][0]

    if overrule_fov is not None:
        fovy = overrule_fov

    position = Vector3(cur_frame_data['location'])
    rotation_eulers = (Vector3(
        [np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) - cor_rotation_eulers) * -1

    position += cor_translation
    rotation = Quaternion.from_eulers(rotation_eulers)

    return Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)


# =============================================================================
# GEOREFERENCING
# =============================================================================

def georeference_polygon(
        polygon_points: List[Tuple[float, float]],
        input_resolution: Resolution,
        tri_mesh: Trimesh,
        camera: Camera,
        x_offset: float,
        y_offset: float,
        z_offset: float,
        add_offsets: bool = True,
        rel_transformer: Optional[Transformer] = None,
        transform_to_target_crs: bool = False
) -> List[Optional[Tuple[float, float, float]]]:
    """Georeference a polygon's points from pixel coordinates to world coordinates."""
    georeferenced_points = []

    for px, py in polygon_points:
        # Create a small bounding box around the point for label_to_world_coordinates
        # Using a 1x1 pixel "box" centered on the point
        point_coords = [px, py, px + 1, py, px + 1, py + 1, px, py + 1]

        world_coordinates = label_to_world_coordinates(
            point_coords,
            input_resolution,
            tri_mesh,
            camera
        )

        if len(world_coordinates) == 0:
            # Point could not be projected - skip it
            georeferenced_points.append(None)
            continue

        # Take the average of the 4 corner points
        xx = world_coordinates[:, 0]
        yy = world_coordinates[:, 1]
        zz = world_coordinates[:, 2]

        if transform_to_target_crs and rel_transformer is not None:
            transformed = rel_transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
            xx = np.array(transformed[0])
            yy = np.array(transformed[1])
            zz = np.array(transformed[2])

        if add_offsets:
            xx = xx + x_offset
            yy = yy + y_offset
            zz = zz + z_offset

        # Use the mean of the projected points
        georeferenced_points.append((float(np.mean(xx)), float(np.mean(yy)), float(np.mean(zz))))

    return georeferenced_points


def georeference_all_polygons(
        polygons: List[Tuple[str, List[Tuple[float, float]]]],
        input_resolution: Resolution,
        tri_mesh: Trimesh,
        camera: Camera,
        x_offset: float,
        y_offset: float,
        z_offset: float,
        add_offsets: bool = True,
        rel_transformer: Optional[Transformer] = None,
        transform_to_target_crs: bool = False
) -> Tuple[List[Tuple[str, List[Optional[Tuple[float, float, float]]]]], int]:
    """Georeference all polygons from a file."""
    georeferenced = []
    total_failed = 0

    for object_type, polygon_points in polygons:
        georef_points = georeference_polygon(
            polygon_points,
            input_resolution,
            tri_mesh,
            camera,
            x_offset,
            y_offset,
            z_offset,
            add_offsets,
            rel_transformer,
            transform_to_target_crs
        )

        # Count failed points
        failed = sum(1 for p in georef_points if p is None)
        total_failed += failed

        georeferenced.append((object_type, georef_points))

    return georeferenced, total_failed


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def load_correction_data(
        correction_folder: str,
        flight_id: str
) -> Tuple[Dict, Dict, float, float, float]:
    """Load correction data for a specific flight."""
    # Load DEM metadata
    parent_id = flight_id[:flight_id.index("_")]
    dem_meta_path = os.path.join(correction_folder, f"{parent_id}_dem_mesh_r2.json")
    with open(dem_meta_path, "r") as f:
        dem_meta = json.load(f)

    x_offset = dem_meta["origin"][0]
    y_offset = dem_meta["origin"][1]
    z_offset = dem_meta["origin"][2]

    # Load matched poses
    poses_path = os.path.join(correction_folder, f"{parent_id}_matched_poses.json")
    with open(poses_path, "r") as f:
        poses = json.load(f)

    # Load correction data
    correction_path = os.path.join(correction_folder, f"{parent_id}_correction.json")
    with open(correction_path, "r") as f:
        correction = json.load(f)

    return poses, correction, x_offset, y_offset, z_offset


def load_dem_mesh(correction_folder: str, flight_id: str) -> Tuple[Any, Trimesh]:
    """Load the DEM mesh for ray casting."""
    parent_id = flight_id[:flight_id.index("_")]
    path_to_dem = os.path.join(correction_folder, f"{parent_id}_dem.glb")
    mesh_data, texture_data = read_gltf(path_to_dem)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    return mesh_data, tri_mesh


def process_polygon_files(
        source_folder: Path,
        target_folder: Path,
        correction_folder: str,
        flight_id: str,
        input_resolution: Resolution = DEFAULT_INPUT_RESOLUTION,
        apply_smoothing: bool = True,
        window_length: int = DEFAULT_WINDOW_LENGTH,
        polyorder: int = DEFAULT_POLYORDER,
        add_offsets: bool = True,
        transform_to_target_crs: bool = False,
        source_crs: int = 4326,
        target_crs: int = 32633,
        additional_corrections_path: Optional[str] = None,
        verbose: bool = True
) -> Dict[str, int]:
    """Process all polygon files in a folder and georeference them."""
    stats = {
        "files_processed": 0,
        "polygons_processed": 0,
        "points_failed": 0,
        "errors": 0
    }

    # Create target folder
    target_folder.mkdir(parents=True, exist_ok=True)

    # Find all polygon files
    polygon_files = list(source_folder.glob("*.txt"))
    if not polygon_files:
        print(f"No polygon files found in {source_folder}")
        return stats

    if verbose:
        print(f"Found {len(polygon_files)} polygon file(s) to process")
        print(f"Flight ID: {flight_id}")
        print("-" * 60)

    # Load correction data
    if verbose:
        print("Loading correction data...")

    poses, correction, x_offset, y_offset, z_offset = load_correction_data(
        correction_folder, flight_id
    )

    # Apply smoothing if requested
    if apply_smoothing:
        if verbose:
            print(f"Applying pose smoothing (window={window_length}, polyorder={polyorder})")
        smooth_pose_positions_savgol(poses, window_length, polyorder)

    # Load additional corrections if provided
    additional_corrections = []
    if additional_corrections_path and os.path.exists(additional_corrections_path):
        with open(additional_corrections_path, "r") as f:
            all_corrections = json.load(f)
        if str(flight_id) in all_corrections.get("corrections", {}):
            additional_corrections = all_corrections["corrections"][str(flight_id)]

    # Get base correction values
    base_translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
    base_rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})

    # Setup CRS transformer if needed
    rel_transformer = None
    if transform_to_target_crs:
        rel_transformer = Transformer.from_crs(
            CRS.from_epsg(source_crs),
            CRS.from_epsg(target_crs)
        )

    # Initialize ModernGL context and load DEM
    ctx = None
    try:
        ctx = make_mgl_context()

        if verbose:
            print("Loading DEM mesh...")
        mesh_data, tri_mesh = load_dem_mesh(correction_folder, flight_id)

        # Process each polygon file
        for i, polygon_file in enumerate(polygon_files, 1):
            if verbose:
                print(f"[{i}/{len(polygon_files)}] Processing: {polygon_file.name}")

            try:
                # 1. Extract frame index from filename
                frame_idx = extract_frame_index(polygon_file.name)

                # 2. Determine Corrections for this specific frame
                translation = base_translation
                rotation = base_rotation

                # Check for frame-specific corrections
                for ac in additional_corrections:
                    if ac.get("start frame", 0) < frame_idx < ac.get("end frame", float('inf')):
                        translation = ac.get('translation', translation)
                        rotation = ac.get('rotation', rotation)
                        break

                # 3. Setup Camera for this specific frame
                cor_rotation_eulers = Vector3(
                    [rotation['x'], rotation['y'], rotation['z']], dtype='f4'
                )
                cor_translation = Vector3(
                    [translation['x'], translation['y'], translation['z']], dtype='f4'
                )

                camera = get_camera_for_frame(poses, frame_idx, cor_rotation_eulers, cor_translation)

                # 4. Read polygons
                polygons = read_polygon_file(polygon_file)

                if not polygons:
                    if verbose:
                        print(f"    No valid polygons found, skipping")
                    continue

                # 5. Georeference all polygons using the frame-specific camera
                georeferenced, failed_points = georeference_all_polygons(
                    polygons,
                    input_resolution,
                    tri_mesh,
                    camera,
                    x_offset,
                    y_offset,
                    z_offset,
                    add_offsets,
                    rel_transformer,
                    transform_to_target_crs
                )

                # Write output file
                output_path = target_folder / polygon_file.name
                header = f"Georeferenced polygons from {polygon_file.name} (flight: {flight_id}, frame: {frame_idx})"
                num_written = write_georeferenced_polygon_file(output_path, georeferenced, header)

                stats["files_processed"] += 1
                stats["polygons_processed"] += num_written
                stats["points_failed"] += failed_points

                if verbose:
                    print(f"    -> Wrote {num_written} polygon(s) to {output_path.name}")
                    if failed_points > 0:
                        print(f"    -> Warning: {failed_points} point(s) could not be georeferenced")

            except ValueError as ve:
                stats["errors"] += 1
                if verbose:
                    print(f"    SKIPPING: {ve}")
            except Exception as e:
                stats["errors"] += 1
                if verbose:
                    print(f"    ERROR: {e}")

    finally:
        if ctx is not None:
            release_all(ctx)

    if verbose:
        print("-" * 60)
        print("Processing complete!")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Polygons georeferenced: {stats['polygons_processed']}")
        print(f"  Points failed: {stats['points_failed']}")
        if stats["errors"] > 0:
            print(f"  Errors: {stats['errors']}")

    return stats


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Georeference SAM3 polygon files using camera poses and DEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -s ./local_polygons -t ./georef_polygons -c ./correction_data -f 223
  %(prog)s -s ./polygons -t ./output -c ./data -f 223 --transform --target-crs 32633
        """
    )

    parser.add_argument(
        "-s", "--source",
        type=Path,
        default=r"Z:\14_1_sam",
        help="Source folder containing local polygon files"
    )

    parser.add_argument(
        "-t", "--target",
        type=Path,
        default=r"Z:\14_1_sam_global",
        help="Target folder for georeferenced polygon files"
    )

    parser.add_argument(
        "-c", "--correction-folder",
        type=str,
        default=r"Z:\correction_data",
        help="Folder containing correction data files (poses, DEM, corrections)"
    )

    parser.add_argument(
        "-f", "--flight-id",
        type=str,
        default="14_1",
        help="Flight identifier (used to find correction files)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Input image width (default: 1024)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Input image height (default: 1024)"
    )

    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable pose smoothing"
    )

    parser.add_argument(
        "--window-length",
        type=int,
        default=DEFAULT_WINDOW_LENGTH,
        help=f"Smoothing window length (default: {DEFAULT_WINDOW_LENGTH})"
    )

    parser.add_argument(
        "--polyorder",
        type=int,
        default=DEFAULT_POLYORDER,
        help=f"Smoothing polynomial order (default: {DEFAULT_POLYORDER})"
    )

    parser.add_argument(
        "--no-offsets",
        action="store_true",
        help="Do not add DEM origin offsets to coordinates"
    )

    parser.add_argument(
        "--transform",
        action="store_true",
        help="Transform coordinates to target CRS"
    )

    parser.add_argument(
        "--source-crs",
        type=int,
        default=4326,
        help="Source EPSG code (default: 4326)"
    )

    parser.add_argument(
        "--target-crs",
        type=int,
        default=32633,
        help="Target EPSG code (default: 32633)"
    )

    parser.add_argument(
        "--additional-corrections",
        type=str,
        default=None,
        help="Path to additional corrections JSON file"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    return parser.parse_args()


def main():
    """Main entry point for command line usage."""
    args = parse_arguments()

    # Validate source folder
    if not args.source.exists():
        print(f"Error: Source folder does not exist: {args.source}")
        sys.exit(1)

    if not args.source.is_dir():
        print(f"Error: Source path is not a directory: {args.source}")
        sys.exit(1)

    # Validate correction folder
    if not os.path.exists(args.correction_folder):
        print(f"Error: Correction folder does not exist: {args.correction_folder}")
        sys.exit(1)

    # Create input resolution
    input_resolution = Resolution(args.width, args.height)

    # Process files
    try:
        stats = process_polygon_files(
            source_folder=args.source,
            target_folder=args.target,
            correction_folder=args.correction_folder,
            flight_id=args.flight_id,
            # frame_idx argument REMOVED
            input_resolution=input_resolution,
            apply_smoothing=not args.no_smoothing,
            window_length=args.window_length,
            polyorder=args.polyorder,
            add_offsets=not args.no_offsets,
            transform_to_target_crs=args.transform,
            source_crs=args.source_crs,
            target_crs=args.target_crs,
            additional_corrections_path=args.additional_corrections,
            verbose=not args.quiet
        )

        if stats.get("errors", 0) > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()