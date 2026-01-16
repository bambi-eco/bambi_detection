"""
DEM Coverage Validator

This script validates whether a DEM (Digital Elevation Model) covers the GPS positions
from an AirData flight log by comparing coordinates in EPSG:32633 (UTM zone 33N).

Usage:
    python validate_dem_coverage.py <air_data_path> <dem_gltf_path>
    python validate_dem_coverage.py --batch <dataset_root_dir>
"""

import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import csv

import numpy as np
from pyproj import Transformer
from trimesh import Trimesh

from src.bambi.srt.srt_parser import SrtParser


@dataclass
class ValidationResult:
    """Results of DEM coverage validation."""
    flight_name: str
    is_valid: bool
    coverage_percentage: float
    points_inside: int
    points_outside: int
    total_points: int

    # Bounding boxes
    dem_min_x: float
    dem_max_x: float
    dem_min_y: float
    dem_max_y: float

    flight_min_x: float
    flight_max_x: float
    flight_min_y: float
    flight_max_y: float

    # Distance metrics
    max_distance_outside: float  # Maximum distance a point is outside the DEM

    error_message: Optional[str] = None


def read_gltf_vertices(gltf_path: Path) -> np.ndarray:
    """
    Read vertices from a GLTF file.

    This is a simplified reader - adjust based on your actual read_gltf implementation.
    """
    import trimesh
    mesh = trimesh.load(str(gltf_path))
    if hasattr(mesh, 'vertices'):
        return np.array(mesh.vertices)
    elif hasattr(mesh, 'geometry'):
        # Scene with multiple geometries
        all_vertices = []
        for geom in mesh.geometry.values():
            all_vertices.append(geom.vertices)
        return np.vstack(all_vertices)
    else:
        raise ValueError(f"Could not extract vertices from {gltf_path}")


def load_dem_bounds(dem_gltf_path: Path, dem_json_path: Path) -> tuple:
    """
    Load DEM and return its bounding box in EPSG:32633 coordinates.

    Returns:
        (min_x, max_x, min_y, max_y, min_z, max_z) in EPSG:32633
    """
    # Load metadata
    with open(dem_json_path, 'r') as f:
        dem_json = json.load(f)

    x_offset = dem_json["origin"][0]
    y_offset = dem_json["origin"][1]
    z_offset = dem_json["origin"][2]

    # Load mesh vertices
    vertices = read_gltf_vertices(dem_gltf_path)

    # Apply offsets to get EPSG:32633 coordinates
    vertices_transformed = vertices.copy()
    vertices_transformed[:, 0] += x_offset
    vertices_transformed[:, 1] += y_offset
    vertices_transformed[:, 2] += z_offset

    # Calculate bounding box
    min_x = vertices_transformed[:, 0].min()
    max_x = vertices_transformed[:, 0].max()
    min_y = vertices_transformed[:, 1].min()
    max_y = vertices_transformed[:, 1].max()
    min_z = vertices_transformed[:, 2].min()
    max_z = vertices_transformed[:, 2].max()

    return min_x, max_x, min_y, max_y, min_z, max_z


def parse_airdata_gps(air_data_path: Path) -> list[tuple[float, float, float]]:
    """
    Parse AirData CSV file and extract GPS coordinates.

    Returns:
        List of (latitude, longitude, altitude) tuples
    """
    coordinates = []

    with open(air_data_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Find the correct column names (AirData uses various naming conventions)
        fieldnames = reader.fieldnames

        # Common column name variations
        lat_cols = ['latitude', 'lat', 'Latitude', 'LAT', 'latitude(degrees)']
        lon_cols = ['longitude', 'lon', 'long', 'Longitude', 'LON', 'longitude(degrees)']
        alt_cols = ['altitude_above_seaLevel(meters)', 'altitude', 'alt', 'Altitude',
                    'altitude(meters)', 'altitude_above_seaLevel(feet)', 'GPS Altitude']

        lat_col = None
        lon_col = None
        alt_col = None

        for col in fieldnames:
            col_lower = col.lower().strip()
            if lat_col is None and any(c.lower() in col_lower for c in lat_cols):
                lat_col = col
            if lon_col is None and any(c.lower() in col_lower for c in lon_cols):
                lon_col = col
            if alt_col is None and any(c.lower() in col_lower for c in alt_cols):
                alt_col = col

        if lat_col is None or lon_col is None:
            raise ValueError(f"Could not find latitude/longitude columns in {air_data_path}. "
                             f"Available columns: {fieldnames}")

        for row in reader:
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                alt = float(row[alt_col]) if alt_col and row.get(alt_col) else 0.0

                # Skip invalid coordinates
                if lat == 0.0 and lon == 0.0:
                    continue
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue

                coordinates.append((lat, lon, alt))
            except (ValueError, KeyError):
                continue

    return coordinates


def gps_to_epsg32633(coordinates: list[tuple[float, float, float]]) -> np.ndarray:
    """
    Convert GPS coordinates (WGS84) to EPSG:32633 (UTM zone 33N).

    Args:
        coordinates: List of (latitude, longitude, altitude) tuples

    Returns:
        numpy array of shape (N, 3) with (x, y, z) in EPSG:32633
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=False)

    result = []
    for lat, lon, alt in coordinates:
        x, y = transformer.transform(lat, lon)
        result.append((x, y, alt))

    return np.array(result)


def calculate_distance_to_bbox(point: tuple[float, float],
                               min_x: float, max_x: float,
                               min_y: float, max_y: float) -> float:
    """
    Calculate the distance from a point to the nearest edge of a bounding box.
    Returns 0 if the point is inside the bounding box.
    """
    x, y = point

    dx = max(min_x - x, 0, x - max_x)
    dy = max(min_y - y, 0, y - max_y)

    return np.sqrt(dx ** 2 + dy ** 2)


def validate_dem_coverage(meta_data_path: Path,
                          dem_gltf_path: Path,
                          dem_json_path: Optional[Path] = None,
                          margin_meters: float = 0.0) -> ValidationResult:
    """
    Validate that a DEM covers the flight path from an AirData log.

    Args:
        meta_data_path: Path to AirData CSV file
        dem_gltf_path: Path to DEM GLTF file
        dem_json_path: Path to DEM metadata JSON file (defaults to same name as GLTF)
        margin_meters: Additional margin around DEM to consider as "covered"

    Returns:
        ValidationResult with detailed coverage information
    """
    flight_name = meta_data_path.parent.name

    # Default JSON path
    if dem_json_path is None:
        dem_json_path = dem_gltf_path.with_suffix('.json')

    # Load DEM bounds
    dem_min_x, dem_max_x, dem_min_y, dem_max_y, _, _ = load_dem_bounds(
        dem_gltf_path, dem_json_path
    )

    # Apply margin
    dem_min_x -= margin_meters
    dem_max_x += margin_meters
    dem_min_y -= margin_meters
    dem_max_y += margin_meters

    # Parse flight GPS coordinates
    if meta_data_path.suffix == '.csv':
        gps_coords = parse_airdata_gps(meta_data_path)
    elif meta_data_path.suffix == '.json':
        gps_coords = []
        with open(meta_data_path) as meta_data_file:
            json_data = json.load(meta_data_file)
            for image in json_data["images"]:
                lat = image["lat"]
                lng = image["lng"]
                alt = image["location"][2]
                gps_coords.append((lat, lng, alt))
    elif meta_data_path.suffix == '.srt':
        srtparser = SrtParser()
        gps_coords = []
        for frame in srtparser.parse_yield(str(meta_data_path)):
            lat = frame.latitude
            lng = frame.longitude
            alt = frame.abs_alt
            gps_coords.append((lat, lng, alt))
    else:
        raise Exception(f"Unsupported file extension: {meta_data_path.suffix}")

    if len(gps_coords) == 0:
        return ValidationResult(
            flight_name=flight_name,
            is_valid=False,
            coverage_percentage=0.0,
            points_inside=0,
            points_outside=0,
            total_points=0,
            dem_min_x=dem_min_x, dem_max_x=dem_max_x,
            dem_min_y=dem_min_y, dem_max_y=dem_max_y,
            flight_min_x=0, flight_max_x=0,
            flight_min_y=0, flight_max_y=0,
            max_distance_outside=0,
            error_message="No valid GPS coordinates found in AirData file"
        )

    # Convert to EPSG:32633
    utm_coords = gps_to_epsg32633(gps_coords)

    # Flight bounding box
    flight_min_x = utm_coords[:, 0].min()
    flight_max_x = utm_coords[:, 0].max()
    flight_min_y = utm_coords[:, 1].min()
    flight_max_y = utm_coords[:, 1].max()

    # Check coverage for each point
    points_inside = 0
    points_outside = 0
    max_distance_outside = 0.0

    for x, y, _ in utm_coords:
        if dem_min_x <= x <= dem_max_x and dem_min_y <= y <= dem_max_y:
            points_inside += 1
        else:
            points_outside += 1
            dist = calculate_distance_to_bbox(
                (x, y), dem_min_x, dem_max_x, dem_min_y, dem_max_y
            )
            max_distance_outside = max(max_distance_outside, dist)

    total_points = points_inside + points_outside
    coverage_percentage = (points_inside / total_points * 100) if total_points > 0 else 0

    # Consider valid if all points are inside (or within margin)
    is_valid = points_outside == 0

    return ValidationResult(
        flight_name=flight_name,
        is_valid=is_valid,
        coverage_percentage=coverage_percentage,
        points_inside=points_inside,
        points_outside=points_outside,
        total_points=total_points,
        dem_min_x=dem_min_x, dem_max_x=dem_max_x,
        dem_min_y=dem_min_y, dem_max_y=dem_max_y,
        flight_min_x=flight_min_x, flight_max_x=flight_max_x,
        flight_min_y=flight_min_y, flight_max_y=flight_max_y,
        max_distance_outside=max_distance_outside
    )

def print_result(result: ValidationResult, verbose: bool = False) -> None:
    """Print validation result in a readable format."""
    status = "✓ VALID" if result.is_valid else "✗ INVALID"

    print(f"\n{'=' * 60}")
    print(f"Flight: {result.flight_name}")
    print(f"Status: {status}")
    print(f"{'=' * 60}")

    if result.error_message:
        print(f"Error: {result.error_message}")
        return

    print(f"Coverage: {result.coverage_percentage:.1f}%")
    print(f"Points inside DEM:  {result.points_inside:,}")
    print(f"Points outside DEM: {result.points_outside:,}")
    print(f"Total points:       {result.total_points:,}")

    if result.points_outside > 0:
        print(f"\nMax distance outside DEM: {result.max_distance_outside:.1f} meters")

    if verbose:
        print(f"\nDEM Bounds (EPSG:32633):")
        print(f"  X: {result.dem_min_x:.1f} to {result.dem_max_x:.1f}")
        print(f"  Y: {result.dem_min_y:.1f} to {result.dem_max_y:.1f}")
        print(f"\nFlight Bounds (EPSG:32633):")
        print(f"  X: {result.flight_min_x:.1f} to {result.flight_max_x:.1f}")
        print(f"  Y: {result.flight_min_y:.1f} to {result.flight_max_y:.1f}")


def find_flight_folders(dataset_root: Path) -> list[tuple[Path, Path]]:
    """
    Find all flight folders containing air_data.csv and DEM files.

    Returns:
        List of (air_data_path, dem_gltf_path) tuples
    """
    flights = []

    # Common patterns for finding files
    air_data_patterns = ['air_data.csv', 'airdata.csv', 'AirData.csv']
    dem_patterns = ['dem_mesh*.gltf', 'dem*.gltf', '*.gltf']

    for folder in dataset_root.rglob('*'):
        if not folder.is_dir():
            continue

        # Look for AirData file
        air_data_path = None
        for pattern in air_data_patterns:
            matches = list(folder.glob(pattern))
            if matches:
                air_data_path = matches[0]
                break

        if air_data_path is None:
            continue

        # Look for DEM GLTF file
        dem_path = None
        for pattern in dem_patterns:
            matches = list(folder.glob(pattern))
            if matches:
                # Prefer dem_mesh files
                for m in matches:
                    if 'dem' in m.name.lower():
                        dem_path = m
                        break
                if dem_path is None and matches:
                    dem_path = matches[0]
                break

        if dem_path is None:
            continue

        # Check if JSON metadata exists
        json_path = dem_path.with_suffix('.json')
        if not json_path.exists():
            continue

        flights.append((air_data_path, dem_path))

    return flights


def batch_validate(dataset_root: Path,
                   margin_meters: float = 0.0,
                   verbose: bool = False) -> list[ValidationResult]:
    """
    Validate all flights in a dataset directory.
    """
    flights = find_flight_folders(dataset_root)

    print(f"Found {len(flights)} flights to validate")

    results = []
    valid_count = 0
    invalid_count = 0

    for air_data_path, dem_path in flights:
        result = validate_dem_coverage(air_data_path, dem_path, margin_meters=margin_meters)
        results.append(result)

        if result.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print_result(result, verbose=verbose)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total flights:   {len(results)}")
    print(f"Valid:           {valid_count}")
    print(f"Invalid:         {invalid_count}")

    if invalid_count > 0:
        print(f"\nInvalid flights:")
        for r in results:
            if not r.is_valid:
                print(f"  - {r.flight_name}: {r.error_message or f'{r.coverage_percentage:.1f}% coverage'}")

    return results

if __name__ == '__main__':
    # result = validate_dem_coverage(
    #     Path(r"C:\D\Projects\alfs_detection\testdata\haag\air_data.csv"),
    #     Path(r"C:\D\Projects\alfs_detection\testdata\haag\dem_mesh_r2.glb"),
    #     margin_meters=0.0
    # )
    # print_result(result)

    parent_folder = Path(r"Z:\correction_data")
    correct_results = {}
    incorrect_results = {}
    glb_files = list(parent_folder.glob("*.glb"))
    for glb_file in glb_files:
        id_ = glb_file.stem.replace("_dem", "")
        print(f"Validating flight {id_}...")
        poses_json = glb_file.with_name(f"{id_}_matched_poses.json")
        dem_json = glb_file.with_name(f"{id_}_dem_mesh_r2.json")

        result = validate_dem_coverage(
            poses_json,
            glb_file,
            dem_json,
            margin_meters=0.0
        )

        if result.is_valid:
            correct_results[id_] = result
        else:
            incorrect_results[id_] = result

    if len(incorrect_results) > 0:
        for key, item in incorrect_results.items():
            print(key)
            print_result(item)
            print("="*60)
    else:
        print("All DEM correct!")