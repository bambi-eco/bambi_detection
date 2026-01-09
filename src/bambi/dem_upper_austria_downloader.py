#!/usr/bin/env python3
"""
Austria DEM Processor - Fully Automated

Downloads, merges, and processes Digital Elevation Model (DGM) data from the
Austrian Federal Office of Metrology and Surveying (BEV) ATOM service.

This version uses the Austria-wide 1m ALS-DTM dataset which:
- Covers all of Austria
- Has 1m resolution (from Airborne Laser Scanning)
- Is organized in 50km x 50km tiles in EPSG:3035
- Is freely available under CC-BY-4.0 license

Data source: https://data.bev.gv.at

Usage:
    python austria_dem_processor.py \\
        --start-lat 48.0995 --start-lon 14.5574 \\
        --end-lat 48.1005 --end-lon 14.5584 \\
        --padding 50 \\
        --output output/dem_mesh

Author: Generated for BAMBI Wildlife Detection project
"""

import argparse
import hashlib
import json
import logging
import os
import re
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import numpy as np

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError:
    print("Error: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)

try:
    from pyproj import Transformer, CRS as PyprojCRS
except ImportError:
    print("Error: pyproj is required. Install with: pip install pyproj")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests is required. Install with: pip install requests")
    sys.exit(1)

try:
    from shapely.geometry import box, mapping
except ImportError:
    print("Error: shapely is required. Install with: pip install shapely")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WGS84_CRS = "EPSG:4326"
BEV_CRS = "EPSG:3035"  # ETRS89-extended / LAEA Europe
DEFAULT_OUTPUT_CRS = "EPSG:32633"  # UTM zone 33N

# BEV ATOM service
BEV_ATOM_SERVICE = "https://data.bev.gv.at/geonetwork/srv/atom/describe/service"
BEV_ATOM_UUID = "208cff7a-c8aa-42fe-bf4f-2b8156e37528"

# Multiple URL patterns to try (in order of preference - newest first)
# Pattern: (date_folder, filename_pattern)
BEV_URL_PATTERNS = [
    # 2023 dataset (most recent)
    ("20230915", "ALS_DTM_CRS3035RES50000mN{north}E{east}.tif"),
    # 2019 dataset (older, may have different tiles)
    ("20190915", "CRS3035RES50000mN{north}E{east}.tif"),
    # 2021 dataset (intermediate)
    ("20210401", "ALS_DTM_CRS3035RES50000mN{north}E{east}.tif"),
]
BEV_DOWNLOAD_BASE = "https://data.bev.gv.at/download/ALS/DTM/"

# Tile parameters (50km x 50km tiles in EPSG:3035)
TILE_SIZE = 50000  # 50km in meters

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "austria_dem"


@dataclass
class BoundingBox:
    """Represents a geographic bounding box."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @classmethod
    def from_points(cls, lat1: float, lon1: float, lat2: float, lon2: float,
                    padding_meters: float = 0) -> 'BoundingBox':
        """Create bounding box from two GPS points with optional padding."""
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)

        if padding_meters > 0:
            lat_center = (min_lat + max_lat) / 2
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))

            lat_padding = padding_meters / meters_per_degree_lat
            lon_padding = padding_meters / meters_per_degree_lon

            min_lat -= lat_padding
            max_lat += lat_padding
            min_lon -= lon_padding
            max_lon += lon_padding

        return cls(min_lat, min_lon, max_lat, max_lon)

    def to_crs(self, target_crs: str) -> Tuple[float, float, float, float]:
        """Convert to projected coordinates (min_x, min_y, max_x, max_y)."""
        transformer = Transformer.from_crs(WGS84_CRS, target_crs, always_xy=True)

        # Transform all four corners to handle projection distortion
        corners = [
            (self.min_lon, self.min_lat),
            (self.max_lon, self.min_lat),
            (self.min_lon, self.max_lat),
            (self.max_lon, self.max_lat)
        ]

        x_coords = []
        y_coords = []
        for lon, lat in corners:
            x, y = transformer.transform(lon, lat)
            x_coords.append(x)
            y_coords.append(y)

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


class BEVTileCalculator:
    """Calculates which BEV tiles are needed for a given bounding box."""

    # Austria approximate bounds in EPSG:3035
    AUSTRIA_BOUNDS = {
        'min_x': 4250000,  # ~9.5° E
        'max_x': 4900000,  # ~17.5° E
        'min_y': 2550000,  # ~46° N
        'max_y': 2900000  # ~49° N
    }

    def __init__(self):
        self.tile_size = TILE_SIZE

    def get_tile_bounds(self, tile_name: str) -> Tuple[float, float, float, float]:
        """
        Parse tile name to get bounds in EPSG:3035.

        Tile names like: CRS3035RES50000mN2700000E4500000
        """
        match = re.search(r'N(\d+)E(\d+)', tile_name)
        if not match:
            return None

        north = int(match.group(1))
        east = int(match.group(2))

        min_x = east
        min_y = north
        max_x = east + self.tile_size
        max_y = north + self.tile_size

        return (min_x, min_y, max_x, max_y)

    def get_tile_name(self, north: int, east: int) -> str:
        """Generate tile identifier from grid coordinates."""
        # This is just the identifier, actual filename varies by dataset version
        return f"N{north}E{east}"

    def get_required_tiles(self, bbox: BoundingBox) -> List[str]:
        """
        Get list of tile names that intersect the bounding box.

        Args:
            bbox: Bounding box in WGS84

        Returns:
            List of tile names
        """
        # Convert bbox to EPSG:3035
        min_x, min_y, max_x, max_y = bbox.to_crs(BEV_CRS)

        logger.debug(f"Bbox in EPSG:3035: {min_x:.0f}, {min_y:.0f} to {max_x:.0f}, {max_y:.0f}")

        # Calculate tile grid indices
        start_east = int(min_x // self.tile_size) * self.tile_size
        end_east = int(max_x // self.tile_size) * self.tile_size
        start_north = int(min_y // self.tile_size) * self.tile_size
        end_north = int(max_y // self.tile_size) * self.tile_size

        tiles = []
        for north in range(start_north, end_north + self.tile_size, self.tile_size):
            for east in range(start_east, end_east + self.tile_size, self.tile_size):
                tile_name = self.get_tile_name(north, east)
                tiles.append(tile_name)

        logger.info(f"Required tiles: {len(tiles)}")
        for tile in tiles:
            logger.debug(f"  {tile}")

        return tiles

    def get_download_urls(self, tile_name: str) -> List[str]:
        """
        Get list of possible download URLs for a tile (tries multiple patterns).

        Returns URLs in order of preference (newest dataset first).
        """
        # Parse the tile identifier
        match = re.search(r'N(\d+)E(\d+)', tile_name)
        if not match:
            return []

        north = match.group(1)
        east = match.group(2)

        urls = []
        for date_folder, filename_pattern in BEV_URL_PATTERNS:
            filename = filename_pattern.format(north=north, east=east)
            url = f"{BEV_DOWNLOAD_BASE}{date_folder}/{filename}"
            urls.append(url)

        return urls

    def get_download_url(self, tile_name: str) -> str:
        """Get the first (preferred) download URL for a tile."""
        urls = self.get_download_urls(tile_name)
        return urls[0] if urls else None


class BEVDownloader:
    """Handles downloading and caching of BEV DEM tiles."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Austria-DEM-Processor/1.0 (Wildlife Research)'
        })
        self.tile_calculator = BEVTileCalculator()

    def _get_cache_path(self, tile_name: str) -> Path:
        """Get the cache path for a tile."""
        return self.cache_dir / f"{tile_name}.tif"

    def download_tile(self, tile_name: str, force: bool = False) -> Optional[Path]:
        """
        Download a single tile.

        Args:
            tile_name: Name of the tile
            force: Force re-download even if cached

        Returns:
            Path to downloaded file or None on failure
        """
        cache_path = self._get_cache_path(tile_name)

        if cache_path.exists() and not force:
            logger.info(f"Using cached: {tile_name}")
            return cache_path

        # Try multiple URL patterns
        urls = self.tile_calculator.get_download_urls(tile_name)

        for url in urls:
            logger.info(f"Trying: {tile_name}")
            logger.debug(f"  URL: {url}")

            try:
                response = self.session.get(url, stream=True, timeout=600)

                if response.status_code == 404:
                    logger.debug(f"  Not found at this URL, trying next...")
                    continue

                response.raise_for_status()

                # Get file size for progress
                total_size = int(response.headers.get('content-length', 0))

                # Download with progress
                downloaded = 0
                with open(cache_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"  Progress: {percent:.1f}%")

                size_mb = cache_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded: {tile_name} ({size_mb:.1f} MB)")
                return cache_path

            except requests.RequestException as e:
                logger.debug(f"  Failed: {e}")
                continue

        # None of the URLs worked
        logger.warning(f"Tile not found at any URL pattern: {tile_name}")
        if cache_path.exists():
            cache_path.unlink()
        return None

    def download_tiles_for_bbox(self, bbox: BoundingBox,
                                force: bool = False) -> List[Path]:
        """
        Download all tiles needed for a bounding box.

        Args:
            bbox: Bounding box in WGS84
            force: Force re-download

        Returns:
            List of paths to downloaded tiles
        """
        tile_names = self.tile_calculator.get_required_tiles(bbox)

        downloaded_tiles = []
        for tile_name in tile_names:
            tile_path = self.download_tile(tile_name, force=force)
            if tile_path:
                downloaded_tiles.append(tile_path)

        if not downloaded_tiles:
            logger.error("No tiles were downloaded successfully")

        return downloaded_tiles


class DEMProcessor:
    """Processes DEM data - merging, clipping, and transforming."""

    def __init__(self, output_crs: str = DEFAULT_OUTPUT_CRS):
        self.output_crs = output_crs

    def merge_and_clip(self, input_files: List[Path], output_file: Path,
                       bbox: BoundingBox) -> Optional[Path]:
        """
        Merge multiple GeoTIFF files and clip to bounding box in one step.

        This is more efficient as it avoids creating large intermediate files.
        """
        if not input_files:
            logger.error("No input files to process")
            return None

        logger.info(f"Processing {len(input_files)} tile(s)...")

        try:
            # Open all files
            src_files = [rasterio.open(f) for f in input_files]

            # Get the source CRS (should be EPSG:3035 for BEV tiles)
            src_crs = src_files[0].crs

            # Convert bbox to source CRS for clipping
            min_x, min_y, max_x, max_y = bbox.to_crs(str(src_crs))
            clip_box = box(min_x, min_y, max_x, max_y)

            logger.info(f"Clip bounds ({src_crs}): {min_x:.1f}, {min_y:.1f} to {max_x:.1f}, {max_y:.1f}")

            if len(src_files) == 1:
                # Single file - just clip
                src = src_files[0]
                out_image, out_transform = mask(
                    src, [mapping(clip_box)], crop=True, all_touched=True
                )
                out_meta = src.meta.copy()
            else:
                # Multiple files - merge first, then clip
                mosaic, mosaic_transform = merge(src_files)

                # Create temporary merged file for clipping
                merged_meta = src_files[0].meta.copy()
                merged_meta.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": mosaic_transform,
                })

                # Write temporary merged file
                temp_merged = output_file.parent / "temp_merged.tif"
                with rasterio.open(temp_merged, 'w', **merged_meta) as dst:
                    dst.write(mosaic)

                # Clip the merged file
                with rasterio.open(temp_merged) as src:
                    out_image, out_transform = mask(
                        src, [mapping(clip_box)], crop=True, all_touched=True
                    )
                    out_meta = src.meta.copy()

                # Clean up temp file
                temp_merged.unlink()

            # Update metadata
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })

            # Write clipped file
            with rasterio.open(output_file, "w", **out_meta) as dst:
                dst.write(out_image)

            # Close source files
            for src in src_files:
                src.close()

            logger.info(f"Clipped to: {output_file} ({out_image.shape[2]}x{out_image.shape[1]})")
            return output_file

        except Exception as e:
            logger.error(f"Failed to process tiles: {e}")
            import traceback
            traceback.print_exc()
            return None

    def reproject_geotiff(self, input_file: Path, output_file: Path,
                          target_crs: Optional[str] = None) -> Optional[Path]:
        """Reproject a GeoTIFF to a different CRS."""
        if target_crs is None:
            target_crs = self.output_crs

        logger.info(f"Reprojecting to {target_crs}...")

        try:
            with rasterio.open(input_file) as src:
                if str(src.crs) == target_crs:
                    logger.info("Already in target CRS, copying...")
                    import shutil
                    shutil.copy2(input_file, output_file)
                    return output_file

                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )

                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'compress': 'lzw'
                })

                with rasterio.open(output_file, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )

                logger.info(f"Reprojected to: {output_file}")
                return output_file

        except Exception as e:
            logger.error(f"Failed to reproject GeoTIFF: {e}")
            return None


class GLTFMeshGenerator:
    """Generates GLTF meshes from DEM data."""

    def __init__(self, simplify_factor: int = 1):
        self.simplify_factor = max(1, simplify_factor)

    def generate_mesh(self, geotiff_path: Path, output_path: Path,
                      metadata_path: Optional[Path] = None) -> bool:
        """Generate a GLTF mesh from a GeoTIFF DEM."""
        logger.info(f"Generating GLTF mesh from {geotiff_path.name}...")

        try:
            with rasterio.open(geotiff_path) as src:
                elevation = src.read(1)

                # Handle nodata
                nodata = src.nodata
                if nodata is not None:
                    elevation = np.where(elevation == nodata, np.nan, elevation)

                # Treat zeros as nodata (edge effect from reprojection)
                elevation = np.where(elevation == 0, np.nan, elevation)

                transform = src.transform
                crs = str(src.crs)
                width = src.width
                height = src.height
                bounds = src.bounds

                # Origin at bottom-left corner
                origin_x = bounds.left
                origin_y = bounds.bottom

                # Get minimum elevation
                valid_elevations = elevation[~np.isnan(elevation)]
                if len(valid_elevations) == 0:
                    logger.error("No valid elevation data")
                    return False

                min_elevation = float(np.nanmin(valid_elevations))
                origin_z = min_elevation

                # Convert origin to WGS84
                transformer = Transformer.from_crs(crs, WGS84_CRS, always_xy=True)
                origin_lon, origin_lat = transformer.transform(origin_x, origin_y)

                # Apply simplification
                if self.simplify_factor > 1:
                    elevation = elevation[::self.simplify_factor, ::self.simplify_factor]
                    mesh_height, mesh_width = elevation.shape
                    pixel_width = transform.a * self.simplify_factor
                    pixel_height = transform.e * self.simplify_factor
                else:
                    mesh_height, mesh_width = elevation.shape
                    pixel_width = transform.a
                    pixel_height = transform.e

                # Generate mesh
                success = self._create_gltf(
                    elevation, output_path,
                    pixel_width, abs(pixel_height),
                    origin_z
                )

                if not success:
                    return False

                # Generate metadata
                if metadata_path is None:
                    metadata_path = output_path.with_suffix('.json')

                # 9-element affine transform
                transform_9 = [
                    transform.a, transform.b, transform.c,
                    transform.d, transform.e, transform.f,
                    0.0, 0.0, 1.0
                ]

                metadata = {
                    "width": width,
                    "height": height,
                    "crs": crs,
                    "transform": transform_9,
                    "origin": [
                        float(origin_x),
                        float(origin_y),
                        float(origin_z)
                    ],
                    "origin_wgs84": {
                        "latitude": float(origin_lat),
                        "longitude": float(origin_lon),
                        "altitude": float(origin_z)
                    }
                }

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                logger.info(f"Created metadata: {metadata_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to generate mesh: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_gltf(self, elevation: np.ndarray, output_path: Path,
                     cell_width: float, cell_height: float,
                     base_elevation: float) -> bool:
        """Create a GLTF file from elevation data."""
        height, width = elevation.shape

        # Replace NaN with base elevation
        elevation = np.nan_to_num(elevation, nan=base_elevation)
        elevation_normalized = elevation - base_elevation

        # Generate vertices
        vertices = []
        for row in range(height):
            for col in range(width):
                x = col * cell_width
                y = (height - 1 - row) * cell_height
                z = float(elevation_normalized[row, col])
                vertices.extend([x, y, z])

        # Generate triangle indices
        indices = []
        for row in range(height - 1):
            for col in range(width - 1):
                i00 = row * width + col
                i10 = row * width + col + 1
                i01 = (row + 1) * width + col
                i11 = (row + 1) * width + col + 1
                indices.extend([i00, i01, i10, i10, i01, i11])

        # Calculate normals
        vertices_array = np.array(vertices).reshape(-1, 3)
        normals = self._calculate_normals(vertices_array, indices)

        return self._write_gltf(
            vertices_array.flatten().tolist(),
            indices,
            normals,
            output_path
        )

    def _calculate_normals(self, vertices: np.ndarray, indices: List[int]) -> List[float]:
        """Calculate per-vertex normals."""
        normals = np.zeros_like(vertices)
        indices_array = np.array(indices).reshape(-1, 3)

        for tri in indices_array:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal /= norm
            for idx in tri:
                normals[idx] += face_normal

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        normals = normals / norms

        return normals.flatten().tolist()

    def _write_gltf(self, vertices: List[float], indices: List[int],
                    normals: List[float], output_path: Path) -> bool:
        """Write GLTF 2.0 binary file."""
        vertices_np = np.array(vertices, dtype=np.float32)
        normals_np = np.array(normals, dtype=np.float32)
        indices_np = np.array(indices, dtype=np.uint32)

        vertices_reshaped = vertices_np.reshape(-1, 3)
        v_min = vertices_reshaped.min(axis=0).tolist()
        v_max = vertices_reshaped.max(axis=0).tolist()

        normals_reshaped = normals_np.reshape(-1, 3)
        n_min = normals_reshaped.min(axis=0).tolist()
        n_max = normals_reshaped.max(axis=0).tolist()

        vertices_bytes = vertices_np.tobytes()
        normals_bytes = normals_np.tobytes()
        indices_bytes = indices_np.tobytes()

        def pad_to_4(data: bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        vertices_bytes_padded = pad_to_4(vertices_bytes)
        normals_bytes_padded = pad_to_4(normals_bytes)
        indices_bytes_padded = pad_to_4(indices_bytes)

        vertices_offset = 0
        normals_offset = len(vertices_bytes_padded)
        indices_offset = normals_offset + len(normals_bytes_padded)

        buffer_data = vertices_bytes_padded + normals_bytes_padded + indices_bytes_padded
        buffer_length = len(buffer_data)

        gltf = {
            "asset": {"version": "2.0", "generator": "Austria-DEM-Processor"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(vertices) // 3, "type": "VEC3", "min": v_min,
                 "max": v_max},
                {"bufferView": 1, "componentType": 5126, "count": len(normals) // 3, "type": "VEC3", "min": n_min,
                 "max": n_max},
                {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR",
                 "min": [int(min(indices))], "max": [int(max(indices))]}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": vertices_offset, "byteLength": len(vertices_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": normals_offset, "byteLength": len(normals_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": indices_offset, "byteLength": len(indices_bytes), "target": 34963}
            ],
            "buffers": [{"byteLength": buffer_length}]
        }

        output_glb = output_path.with_suffix('.glb')

        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes_padded = json_bytes + b' ' * json_padding

        glb_length = 12 + 8 + len(json_bytes_padded) + 8 + buffer_length

        with open(output_glb, 'wb') as f:
            f.write(b'glTF')
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<I', glb_length))
            f.write(struct.pack('<I', len(json_bytes_padded)))
            f.write(b'JSON')
            f.write(json_bytes_padded)
            f.write(struct.pack('<I', buffer_length))
            f.write(b'BIN\x00')
            f.write(buffer_data)

        logger.info(f"Created GLTF mesh: {output_glb}")
        logger.info(f"  Vertices: {len(vertices) // 3}")
        logger.info(f"  Triangles: {len(indices) // 3}")
        logger.info(f"  File size: {output_glb.stat().st_size / 1024 / 1024:.2f} MB")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fully automated Austria DEM processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process area between two GPS points with 50m padding
  python austria_dem_processor.py \\
      --start-lat 48.10 --start-lon 14.55 \\
      --end-lat 48.11 --end-lon 14.56 \\
      --padding 50 \\
      --output dem_output

  # Full resolution (1m) - may create large files
  python austria_dem_processor.py \\
      --start-lat 48.10 --start-lon 14.55 \\
      --end-lat 48.11 --end-lon 14.56 \\
      --simplify 1 \\
      --output dem_output

Data source: Austrian Federal Office (BEV) - 1m ALS-DTM
License: CC-BY-4.0
        """
    )

    # Required arguments
    parser.add_argument('--start-lat', type=float, default="48.0995525449297",
                        help='Start point latitude (WGS84)')
    parser.add_argument('--start-lon', type=float, default="14.5573984876557",
                        help='Start point longitude (WGS84)')
    parser.add_argument('--end-lat', type=float, default="48.1295525449297",
                        help='End point latitude (WGS84)')
    parser.add_argument('--end-lon', type=float, default="14.5973984876557",
                        help='End point longitude (WGS84)')
    parser.add_argument('--padding', type=float, default=30,
                        help='Padding in meters around bounding box')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output base path (without extension)')

    # Optional arguments
    parser.add_argument('--output-crs', default=DEFAULT_OUTPUT_CRS,
                        help=f'Output CRS (default: {DEFAULT_OUTPUT_CRS})')
    parser.add_argument('--simplify', type=int, default=2,
                        help='Mesh simplification factor (1=full 1m, 2=2m, etc.)')
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                        help=f'Cache directory (default: {DEFAULT_CACHE_DIR})')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download even if cached')
    parser.add_argument('--verbose', '-v', action='store_false',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create bounding box
    bbox = BoundingBox.from_points(
        args.start_lat, args.start_lon,
        args.end_lat, args.end_lon,
        args.padding
    )

    logger.info("=" * 60)
    logger.info("Austria DEM Processor - Fully Automated")
    logger.info("=" * 60)
    logger.info(f"Bounding box (WGS84):")
    logger.info(f"  SW: {bbox.min_lat:.6f}, {bbox.min_lon:.6f}")
    logger.info(f"  NE: {bbox.max_lat:.6f}, {bbox.max_lon:.6f}")

    # Initialize components
    downloader = BEVDownloader(cache_dir=args.cache_dir)
    processor = DEMProcessor(output_crs=args.output_crs)
    mesh_generator = GLTFMeshGenerator(simplify_factor=args.simplify)

    # Step 1: Download required tiles
    logger.info("-" * 60)
    logger.info("Step 1: Downloading required tiles...")
    tile_paths = downloader.download_tiles_for_bbox(bbox, force=args.force_download)

    if not tile_paths:
        logger.error("No tiles downloaded - area may be outside Austria")
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Step 2: Merge and clip
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        logger.info("-" * 60)
        logger.info("Step 2: Merging and clipping tiles...")
        clipped_file = temp_path / "clipped.tif"
        result = processor.merge_and_clip(tile_paths, clipped_file, bbox)
        if not result:
            sys.exit(1)

        # Step 3: Reproject
        logger.info("-" * 60)
        logger.info("Step 3: Reprojecting to output CRS...")
        reprojected_file = temp_path / "reprojected.tif"
        result = processor.reproject_geotiff(clipped_file, reprojected_file)
        if not result:
            sys.exit(1)

        # Step 4: Copy final GeoTIFF
        final_geotiff = args.output.with_suffix('.tif')
        import shutil
        shutil.copy2(reprojected_file, final_geotiff)
        logger.info(f"Created output GeoTIFF: {final_geotiff}")

        # Step 5: Generate mesh
        logger.info("-" * 60)
        logger.info("Step 4: Generating GLTF mesh...")
        mesh_file = args.output.with_suffix('.glb')
        metadata_file = args.output.with_suffix('.json')

        success = mesh_generator.generate_mesh(
            final_geotiff, mesh_file, metadata_file
        )

        if not success:
            logger.error("Failed to generate mesh")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"  GeoTIFF:  {final_geotiff}")
    logger.info(f"  Mesh:     {mesh_file}")
    logger.info(f"  Metadata: {metadata_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()