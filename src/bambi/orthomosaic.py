"""
Orthomosaic Generation Module

Creates a seamless orthomosaic from multiple overlapping shots by:
1. Computing a global bounding box covering all shots
2. Setting up a single orthographic camera covering the entire area
3. Projecting all shots onto this global canvas with integral blending
4. Optionally tiling for very high resolution outputs
"""

import os
import json
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path

import cv2
import numpy as np
from moderngl import Context
from pyrr import Quaternion, Vector3
from trimesh import Trimesh

# Optional GeoTIFF support
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("rasterio not installed - GeoTIFF export will not be available. "
                    "Install with: pip install rasterio")

from alfspy.core.geo.aabb import AABB
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering import Resolution, Camera, CtxShot, RenderResultMode, TextureData
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.render.render import (
    make_mgl_context, read_gltf, process_render_data,
    make_shot_loader, release_all
)


from enum import Enum


class BlendMode(Enum):
    """Blending mode for combining multiple shots."""
    INTEGRAL = "integral"      # Average all overlapping pixels (light field style)
    FIRST = "first"            # First-write-wins: keep first valid pixel
    LAST = "last"              # Last-write-wins: overwrite with each new view
    CENTER = "center"          # Prioritize shots closer to output center


@dataclass
class FrameFilter:
    """
    Filter for selecting which frames to include.

    Examples:
        - FrameFilter(start=0, end=100)  -> frames 0-100 (inclusive)
        - FrameFilter(start=60)          -> frames 60 and onwards
        - FrameFilter(end=50)            -> frames 0-50
        - FrameFilter()                  -> all frames (no filtering)
    """
    start: Optional[int] = None  # First frame index to include (inclusive)
    end: Optional[int] = None    # Last frame index to include (inclusive)

    def is_valid(self, frame_idx: int) -> bool:
        """Check if a frame index passes the filter."""
        if self.start is not None and frame_idx < self.start:
            return False
        if self.end is not None and frame_idx > self.end:
            return False
        return True

    def __str__(self) -> str:
        if self.start is None and self.end is None:
            return "all frames"
        elif self.start is None:
            return f"frames <= {self.end}"
        elif self.end is None:
            return f"frames >= {self.start}"
        else:
            return f"frames {self.start} to {self.end}"


@dataclass
class OrthomosaicSettings:
    """Configuration for orthomosaic generation."""
    # Output resolution (meters per pixel)
    ground_resolution: float = 0.05  # 5cm per pixel

    # Maximum tile size in pixels (to handle memory constraints)
    max_tile_size: int = 8192

    # Overlap between tiles (in pixels) for seamless blending
    tile_overlap: int = 128

    # Minimum number of overlapping shots to consider a pixel valid
    alpha_threshold: float = 0.5

    # Auto contrast adjustment
    auto_contrast: bool = True

    # Camera settings
    near_clipping: float = 0.1
    far_clipping: float = 10000.0
    camera_height: float = 100.0  # Height above the highest point

    # Output format
    output_format: str = 'tif'  # 'tif', 'png', 'jpg'
    jpeg_quality: int = 95

    # Frame filtering
    frame_filter: Optional[FrameFilter] = None

    # Default FOV if not specified in poses
    default_fovy: float = 60.0

    # Blending mode for combining multiple shots
    # - INTEGRAL: Average all overlapping pixels (light field style, smooth but can be blurry)
    # - FIRST: First-write-wins, keep first valid pixel (sharp, no blending)
    # - LAST: Last-write-wins, overwrite with each new view
    # - CENTER: Prioritize shots whose camera is closer to the pixel location
    blend_mode: BlendMode = BlendMode.INTEGRAL

    # GeoTIFF settings
    # CRS can be an EPSG code (int), WKT string, or proj4 string
    # Common examples:
    #   - EPSG:4326 (WGS84 lat/lon)
    #   - EPSG:32632 (UTM zone 32N)
    #   - EPSG:31254 (Austria GK West)
    #   - EPSG:31255 (Austria GK Central)
    #   - EPSG:31256 (Austria GK East)
    crs: Optional[Union[int, str]] = None  # e.g., 32632 for UTM 32N, or "EPSG:4326"

    # Enable GeoTIFF output (requires rasterio)
    geotiff: bool = True

    # GeoTIFF compression: 'lzw', 'deflate', 'packbits', 'jpeg', None
    geotiff_compression: Optional[str] = 'lzw'

    # Create overviews (pyramids) for faster viewing at lower zoom levels
    create_overviews: bool = True
    overview_levels: List[int] = None  # Default: [2, 4, 8, 16]

    # Coordinate offsets - use if your DEM is in a local coordinate system
    # These values are ADDED to the computed bounds to get real-world coordinates
    # Example: If your DEM origin is at UTM (456000, 5234000), set:
    #   coord_offset_x = 456000
    #   coord_offset_y = 5234000
    coord_offset_x: float = 0.0
    coord_offset_y: float = 0.0

    # Crop output to minimal area containing projected pixels
    # Removes empty borders around the orthomosaic
    crop_to_content: bool = False

    # Manual coordinate adjustment (applied AFTER the origin offset)
    # Use this to fine-tune the georeferencing if there's a systematic offset
    # Positive values shift the output east/north
    manual_offset_x: float = 0.0
    manual_offset_y: float = 0.0

    def __post_init__(self):
        if self.overview_levels is None:
            self.overview_levels = [2, 4, 8, 16]


def compute_global_bounds(
        shots: List[CtxShot],
        mesh_aabb: AABB,
        padding: float = 10.0
) -> Tuple[float, float, float, float]:
    """
    Compute the global bounding box that encompasses all shot footprints.

    Returns: (min_x, min_y, max_x, max_y) in world coordinates
    """
    # Get shot positions
    shot_positions = np.array([shot.camera.transform.position for shot in shots])

    # Compute AABB of shot positions
    shots_aabb = get_aabb(shot_positions)

    # Access AABB bounds via p_min and p_max
    shots_min_x = float(shots_aabb.p_min.x)
    shots_min_y = float(shots_aabb.p_min.y)
    shots_max_x = float(shots_aabb.p_max.x)
    shots_max_y = float(shots_aabb.p_max.y)

    mesh_min_x = float(mesh_aabb.p_min.x)
    mesh_min_y = float(mesh_aabb.p_min.y)
    mesh_max_x = float(mesh_aabb.p_max.x)
    mesh_max_y = float(mesh_aabb.p_max.y)

    # Combine with mesh AABB (the actual terrain)
    min_x = min(shots_min_x, mesh_min_x) - padding
    min_y = min(shots_min_y, mesh_min_y) - padding
    max_x = max(shots_max_x, mesh_max_x) + padding
    max_y = max(shots_max_y, mesh_max_y) + padding

    return min_x, min_y, max_x, max_y


def create_global_orthographic_camera(
        bounds: Tuple[float, float, float, float],
        mesh_aabb: AABB,
        settings: OrthomosaicSettings
) -> Camera:
    """
    Create an orthographic camera that covers the entire survey area.
    The camera looks straight down (negative Z direction).

    Note: The camera is set up with flipped Y-axis so the rendered output
    matches GeoTIFF convention (row 0 = north/max_y).
    """
    min_x, min_y, max_x, max_y = bounds
    max_z = float(mesh_aabb.p_max.z)

    # Compute center position
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = max_z + settings.camera_height

    # Compute orthographic size (width, height in world units)
    ortho_width = max_x - min_x
    # Negative height flips Y-axis so row 0 = max_y (north), matching GeoTIFF convention
    ortho_height = -(max_y - min_y)

    # Camera rotation: looking straight down
    # Default camera looks along -Z, which is what we want for top-down view
    rotation = Quaternion()  # Identity = looking down -Z

    camera = Camera(
        orthogonal=True,
        orthogonal_size=(ortho_width, ortho_height),
        position=Vector3([center_x, center_y, center_z], dtype='f4'),
        rotation=rotation,
        near=settings.near_clipping,
        far=settings.far_clipping
    )

    return camera


def compute_output_resolution(
        bounds: Tuple[float, float, float, float],
        ground_resolution: float
) -> Resolution:
    """
    Compute the output resolution based on bounds and desired ground resolution.
    """
    min_x, min_y, max_x, max_y = bounds

    width_meters = max_x - min_x
    height_meters = max_y - min_y

    width_pixels = int(math.ceil(width_meters / ground_resolution))
    height_pixels = int(math.ceil(height_meters / ground_resolution))

    return Resolution(width_pixels, height_pixels)


def generate_tiles(
        total_resolution: Resolution,
        max_tile_size: int,
        overlap: int
) -> List[Tuple[int, int, int, int]]:
    """
    Generate tile definitions for processing large orthomosaics.

    Returns list of (x_start, y_start, width, height) tuples.
    """
    tiles = []

    effective_tile_size = max_tile_size - overlap

    for y in range(0, total_resolution.height, effective_tile_size):
        for x in range(0, total_resolution.width, effective_tile_size):
            tile_width = min(max_tile_size, total_resolution.width - x + overlap)
            tile_height = min(max_tile_size, total_resolution.height - y + overlap)

            # Ensure minimum size
            tile_width = min(tile_width, total_resolution.width - x)
            tile_height = min(tile_height, total_resolution.height - y)

            tiles.append((x, y, tile_width, tile_height))

    return tiles


def create_tile_camera(
        global_camera: Camera,
        global_bounds: Tuple[float, float, float, float],
        global_resolution: Resolution,
        tile: Tuple[int, int, int, int]
) -> Camera:
    """
    Create a camera for rendering a specific tile.
    """
    x_start, y_start, tile_width, tile_height = tile
    min_x, min_y, max_x, max_y = global_bounds

    # Convert pixel coordinates to world coordinates
    pixel_to_world_x = (max_x - min_x) / global_resolution.width
    pixel_to_world_y = (max_y - min_y) / global_resolution.height

    # Tile bounds in world coordinates
    tile_min_x = min_x + x_start * pixel_to_world_x
    tile_max_x = min_x + (x_start + tile_width) * pixel_to_world_x
    tile_max_y = max_y - y_start * pixel_to_world_y
    tile_min_y = max_y - (y_start + tile_height) * pixel_to_world_y

    # Tile center and size
    tile_center_x = (tile_min_x + tile_max_x) / 2.0
    tile_center_y = (tile_min_y + tile_max_y) / 2.0
    tile_ortho_width = tile_max_x - tile_min_x
    tile_ortho_height = tile_max_y - tile_min_y

    return Camera(
        orthogonal=True,
        orthogonal_size=(tile_ortho_width, tile_ortho_height),
        position=Vector3([
            tile_center_x,
            tile_center_y,
            global_camera.transform.position.z
        ], dtype='f4'),
        rotation=global_camera.transform.rotation,
        near=global_camera.near,
        far=global_camera.far
    )



# ============================================================================
# DEM Metadata Loading
# ============================================================================

@dataclass
class DEMMetadata:
    """Metadata from the DEM JSON file containing georeferencing information."""
    crs: str                           # e.g., "EPSG:32633"
    origin_x: float                    # UTM Easting of local origin
    origin_y: float                    # UTM Northing of local origin
    origin_z: float                    # Altitude of local origin
    origin_wgs84_lat: Optional[float] = None
    origin_wgs84_lon: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    transform: Optional[List[float]] = None
    # Transform-based origin (from the affine transform, if different)
    transform_origin_x: Optional[float] = None
    transform_origin_y: Optional[float] = None

    def get_epsg_code(self) -> Optional[int]:
        """Extract EPSG code as integer from CRS string."""
        if self.crs and self.crs.upper().startswith("EPSG:"):
            try:
                return int(self.crs.split(":")[1])
            except (ValueError, IndexError):
                return None
        return None


def load_dem_metadata(dem_metadata_file: str) -> Optional[DEMMetadata]:
    """
    Load DEM metadata from JSON file.

    The JSON file should contain:
    - origin: [x, y, z] - UTM coordinates of the local origin
    - crs: "EPSG:XXXXX" - Coordinate reference system
    - Optionally: origin_wgs84, width, height, transform

    Args:
        dem_metadata_file: Path to the DEM metadata JSON file

    Returns:
        DEMMetadata object or None if file doesn't exist or is invalid
    """
    if not dem_metadata_file or not os.path.exists(dem_metadata_file):
        logging.warning(f"DEM metadata file not found: {dem_metadata_file}")
        return None

    try:
        with open(dem_metadata_file, 'r') as f:
            data = json.load(f)

        origin = data.get("origin", [0, 0, 0])
        crs = data.get("crs", "")
        transform = data.get("transform")

        origin_wgs84 = data.get("origin_wgs84", {})

        # Extract transform origin if available
        # Transform format: [pixel_size_x, rot_x, origin_x, rot_y, pixel_size_y, origin_y, 0, 0, 1]
        transform_origin_x = None
        transform_origin_y = None
        if transform and len(transform) >= 6:
            transform_origin_x = float(transform[2])
            transform_origin_y = float(transform[5])

        metadata = DEMMetadata(
            crs=crs,
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            origin_z=float(origin[2]) if len(origin) > 2 else 0.0,
            origin_wgs84_lat=origin_wgs84.get("latitude"),
            origin_wgs84_lon=origin_wgs84.get("longitude"),
            width=data.get("width"),
            height=data.get("height"),
            transform=transform,
            transform_origin_x=transform_origin_x,
            transform_origin_y=transform_origin_y,
        )

        logging.info(f"Loaded DEM metadata from {dem_metadata_file}")
        logging.info(f"  CRS: {metadata.crs}")
        logging.info(f"  Origin (UTM): X={metadata.origin_x:.3f}, Y={metadata.origin_y:.3f}, Z={metadata.origin_z:.3f}")
        if metadata.origin_wgs84_lat and metadata.origin_wgs84_lon:
            logging.info(f"  Origin (WGS84): {metadata.origin_wgs84_lat:.6f}°N, {metadata.origin_wgs84_lon:.6f}°E")
        if transform_origin_x is not None and transform_origin_y is not None:
            logging.info(f"  Transform origin: X={transform_origin_x:.3f}, Y={transform_origin_y:.3f}")
            diff_x = transform_origin_x - metadata.origin_x
            diff_y = transform_origin_y - metadata.origin_y
            logging.info(f"  Difference (transform - origin): dX={diff_x:.2f}m, dY={diff_y:.2f}m")

        return metadata

    except Exception as e:
        logging.error(f"Failed to load DEM metadata from {dem_metadata_file}: {e}")
        return None


# ============================================================================
# Correction Handling
# ============================================================================

def load_corrections_data(
    correction_matrix_file: Optional[str] = None,
    onefile_corrections_file: Optional[str] = None,
    flight_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load correction data from files.

    Args:
        correction_matrix_file: Path to the default/global correction JSON file
        onefile_corrections_file: Path to the unified corrections file with frame-specific corrections
        flight_key: Flight key to look up in the onefile corrections (if applicable)

    Returns:
        Dictionary with "default" and "corrections" keys
    """
    corrections_data = {
        "default": {
            "translation": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0}
        },
        "corrections": []
    }

    # Load global/default correction
    if correction_matrix_file and os.path.exists(correction_matrix_file):
        logging.info(f"Loading default correction from: {correction_matrix_file}")
        with open(correction_matrix_file, 'r') as f:
            corrections_data["default"] = json.load(f)

    # Load frame-specific corrections
    if onefile_corrections_file and os.path.exists(onefile_corrections_file):
        logging.info(f"Loading frame corrections from: {onefile_corrections_file}")
        with open(onefile_corrections_file, 'r') as f:
            all_corrections = json.load(f)

            # Check if this is a per-flight file or a unified file
            if "corrections" in all_corrections:
                if flight_key and isinstance(all_corrections["corrections"], dict):
                    # Unified file with flight keys
                    flight_corrections = all_corrections["corrections"].get(str(flight_key), [])
                    corrections_data["corrections"] = flight_corrections
                elif isinstance(all_corrections["corrections"], list):
                    # Direct list of corrections
                    corrections_data["corrections"] = all_corrections["corrections"]

    return corrections_data


def get_frame_correction(
    corrections_data: Dict[str, Any],
    frame_idx: int,
    central_frame_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get the correction for a specific frame from the corrections data.

    Args:
        corrections_data: Dictionary with "default" and "corrections" keys
        frame_idx: The frame index to get correction for
        central_frame_idx: Optional central frame index for fallback

    Returns:
        Correction dictionary with "translation" and "rotation" keys
    """
    central_correction = None

    for correction in corrections_data.get("corrections", []):
        start_frame = correction.get("start frame", correction.get("start_frame", -1))
        end_frame = correction.get("end frame", correction.get("end_frame", -1))

        if start_frame <= frame_idx <= end_frame:
            return correction

        if central_frame_idx is not None and start_frame <= central_frame_idx <= end_frame:
            central_correction = correction

    if central_correction is not None:
        return central_correction

    return corrections_data["default"]


def correction_dict_to_transform(correction_dict: Dict[str, Any]) -> Tuple[Transform, Vector3]:
    """
    Convert a correction dictionary to a Transform object.

    Returns:
        Tuple of (Transform, rotation_eulers as Vector3)
    """
    translation = correction_dict.get('translation', {'x': 0, 'y': 0, 'z': 0})
    cor_translation = Vector3([
        translation.get('x', 0),
        translation.get('y', 0),
        translation.get('z', 0)
    ], dtype='f4')

    rotation = correction_dict.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    cor_rotation_eulers = Vector3([
        rotation.get('x', 0),
        rotation.get('y', 0),
        rotation.get('z', 0)
    ], dtype='f4')
    cor_quat = Quaternion.from_eulers(cor_rotation_eulers)

    return Transform(cor_translation, cor_quat), cor_rotation_eulers


# ============================================================================
# GeoTIFF Output
# ============================================================================

def save_world_file(
    output_file: str,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int
) -> str:
    """
    Save a world file (.tfw/.pgw/.jgw) for georeferencing.

    World file format (6 lines):
    1. Pixel size in X direction (positive)
    2. Rotation about Y axis (0)
    3. Rotation about X axis (0)
    4. Pixel size in Y direction (negative = Y decreases down image rows)
    5. X coordinate of center of upper-left pixel
    6. Y coordinate of center of upper-left pixel
    """
    min_x, min_y, max_x, max_y = bounds

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = -(max_y - min_y) / height  # Negative: Y decreases as row increases

    # Upper-left pixel (row 0, col 0) center
    origin_x = min_x + pixel_size_x / 2
    origin_y = max_y + pixel_size_y / 2  # max_y minus half pixel

    # Determine world file extension based on image extension
    base, ext = os.path.splitext(output_file)
    ext_lower = ext.lower()
    if ext_lower in ['.tif', '.tiff']:
        world_ext = '.tfw'
    elif ext_lower == '.png':
        world_ext = '.pgw'
    elif ext_lower in ['.jpg', '.jpeg']:
        world_ext = '.jgw'
    else:
        world_ext = '.wld'

    world_file = base + world_ext

    with open(world_file, 'w') as f:
        f.write(f"{pixel_size_x:.10f}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{pixel_size_y:.10f}\n")
        f.write(f"{origin_x:.10f}\n")
        f.write(f"{origin_y:.10f}\n")

    logging.info(f"World file saved: {world_file}")
    return world_file


def save_prj_file(output_file: str, crs: Union[int, str]) -> Optional[str]:
    """
    Save a .prj file with the CRS definition (WKT format).
    This helps GIS software identify the coordinate system.
    """
    if not HAS_RASTERIO:
        return None

    try:
        if isinstance(crs, int):
            rasterio_crs = CRS.from_epsg(crs)
        elif isinstance(crs, str):
            if crs.upper().startswith("EPSG:"):
                rasterio_crs = CRS.from_epsg(int(crs.split(":")[1]))
            else:
                rasterio_crs = CRS.from_string(crs)
        else:
            return None

        base, _ = os.path.splitext(output_file)
        prj_file = base + '.prj'

        with open(prj_file, 'w') as f:
            f.write(rasterio_crs.to_wkt())

        logging.info(f"PRJ file saved: {prj_file}")
        return prj_file

    except Exception as e:
        logging.warning(f"Could not create PRJ file: {e}")
        return None


def save_geotiff(
    image: np.ndarray,
    output_file: str,
    bounds: Tuple[float, float, float, float],
    crs: Optional[Union[int, str]] = None,
    compression: Optional[str] = 'lzw',
    create_overviews: bool = True,
    overview_levels: List[int] = None
) -> bool:
    """
    Save an image as a GeoTIFF with georeferencing.

    Args:
        image: RGBA or RGB numpy array (height, width, channels)
        output_file: Output file path
        bounds: World coordinate bounds (min_x, min_y, max_x, max_y)
        crs: Coordinate reference system (EPSG code or string)
        compression: Compression type ('lzw', 'deflate', 'packbits', 'jpeg', None)
        create_overviews: Whether to create overview pyramids
        overview_levels: Overview levels, default [2, 4, 8, 16]

    Returns:
        True if successful, False otherwise
    """
    if not HAS_RASTERIO:
        logging.error("rasterio is not installed. Cannot save GeoTIFF.")
        logging.error("Install with: pip install rasterio")
        return False

    if overview_levels is None:
        overview_levels = [2, 4, 8, 16]

    min_x, min_y, max_x, max_y = bounds
    height, width = image.shape[:2]

    # Log coordinate information for debugging
    logging.info("=" * 60)
    logging.info("GeoTIFF Coordinate Information:")
    logging.info(f"  Image size: {width} x {height} pixels")
    logging.info(f"  World bounds:")
    logging.info(f"    Min X (West):  {min_x:.6f}")
    logging.info(f"    Max X (East):  {max_x:.6f}")
    logging.info(f"    Min Y (South): {min_y:.6f}")
    logging.info(f"    Max Y (North): {max_y:.6f}")
    logging.info(f"  Extent: {max_x - min_x:.2f} x {max_y - min_y:.2f} world units")
    logging.info(f"  Pixel size: {(max_x - min_x) / width:.6f} x {(max_y - min_y) / height:.6f}")

    # Determine number of bands
    if len(image.shape) == 2:
        count = 1
        image = image[:, :, np.newaxis]
    else:
        count = image.shape[2]

    # Create the geotransform from bounds
    # Standard: from_bounds(west, south, east, north, width, height)
    # Row 0 = north (max_y), rows go southward
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    logging.info(f"  Affine transform: {transform}")

    # Setup CRS
    rasterio_crs = None
    if crs is None:
        logging.warning("=" * 60)
        logging.warning("WARNING: No CRS specified!")
        logging.warning("The GeoTIFF will have coordinates but no coordinate system definition.")
        logging.warning("QGIS/ArcGIS may not display it correctly.")
        logging.warning("")
        logging.warning("To fix this, specify the CRS that matches your DEM's coordinate system:")
        logging.warning("  - For UTM Zone 32N: crs=32632")
        logging.warning("  - For UTM Zone 33N: crs=32633")
        logging.warning("  - For Austria GK West: crs=31254")
        logging.warning("  - For Austria GK Central: crs=31255")
        logging.warning("  - For Austria GK East: crs=31256")
        logging.warning("  - For Austria Lambert: crs=31287")
        logging.warning("=" * 60)
    else:
        try:
            if isinstance(crs, int):
                rasterio_crs = CRS.from_epsg(crs)
            elif isinstance(crs, str):
                if crs.upper().startswith("EPSG:"):
                    rasterio_crs = CRS.from_epsg(int(crs.split(":")[1]))
                else:
                    rasterio_crs = CRS.from_string(crs)
            logging.info(f"  CRS: {rasterio_crs}")
        except Exception as e:
            logging.error(f"Failed to parse CRS '{crs}': {e}")
            rasterio_crs = None

    logging.info("=" * 60)

    # Setup profile
    profile = {
        'driver': 'GTiff',
        'dtype': image.dtype,
        'width': width,
        'height': height,
        'count': count,
        'transform': transform,
    }

    if compression:
        profile['compress'] = compression

    if rasterio_crs is not None:
        profile['crs'] = rasterio_crs

    # Add tiling for large images (better for GIS viewing)
    if width > 256 and height > 256:
        profile['tiled'] = True
        profile['blockxsize'] = 256
        profile['blockysize'] = 256

    try:
        logging.info(f"Writing GeoTIFF to {output_file}")
        with rasterio.open(output_file, 'w', **profile) as dst:
            # Rasterio expects bands in (bands, height, width) order
            for i in range(count):
                dst.write(image[:, :, i], i + 1)

            # Set band descriptions
            if count == 4:
                dst.descriptions = ('Red', 'Green', 'Blue', 'Alpha')
            elif count == 3:
                dst.descriptions = ('Red', 'Green', 'Blue')

            # Create overviews (pyramids) for faster viewing
            if create_overviews and overview_levels:
                logging.info(f"Creating overviews: {overview_levels}")
                dst.build_overviews(overview_levels, rasterio.enums.Resampling.average)
                dst.update_tags(ns='rio_overview', resampling='average')

        logging.info(f"GeoTIFF saved successfully: {output_file}")

        # Also save world file as backup
        save_world_file(output_file, bounds, width, height)

        # Save PRJ file if we have a CRS
        if rasterio_crs is not None:
            save_prj_file(output_file, crs)

        return True

    except Exception as e:
        logging.error(f"Failed to save GeoTIFF: {e}")
        import traceback
        traceback.print_exc()
        return False


def crop_to_content(
    image: np.ndarray,
    bounds: Tuple[float, float, float, float],
    padding: int = 0
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Crop an image to the minimal bounding box containing non-empty pixels.

    Args:
        image: RGBA numpy array (height, width, 4)
        bounds: World coordinate bounds (min_x, min_y, max_x, max_y)
        padding: Optional padding in pixels to add around the content

    Returns:
        Tuple of (cropped_image, new_bounds)
    """
    min_x, min_y, max_x, max_y = bounds
    height, width = image.shape[:2]

    # Find non-empty pixels (alpha > 0)
    alpha = image[:, :, 3]
    non_empty = alpha > 0

    if not np.any(non_empty):
        logging.warning("No content found in image - returning original")
        return image, bounds

    # Find bounding box of non-empty pixels
    rows = np.any(non_empty, axis=1)
    cols = np.any(non_empty, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Apply padding
    row_min = max(0, row_min - padding)
    row_max = min(height - 1, row_max + padding)
    col_min = max(0, col_min - padding)
    col_max = min(width - 1, col_max + padding)

    # Crop image
    cropped = image[row_min:row_max + 1, col_min:col_max + 1]

    # Calculate new bounds
    # Since we swap Y in GeoTIFF (from_bounds uses max_y, min_y),
    # row 0 = max_y (north), row N = min_y (south)
    # So row_min is at high Y, row_max is at low Y
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    new_min_x = min_x + col_min * pixel_size_x
    new_max_x = min_x + (col_max + 1) * pixel_size_x

    # row_min is at top (max_y side), row_max is at bottom (min_y side)
    new_max_y = max_y - row_min * pixel_size_y
    new_min_y = max_y - (row_max + 1) * pixel_size_y

    new_bounds = (new_min_x, new_min_y, new_max_x, new_max_y)

    logging.info(f"Cropped image from {width}x{height} to {cropped.shape[1]}x{cropped.shape[0]}")
    logging.info(f"  Pixel range: X [{col_min}, {col_max}], Y [{row_min}, {row_max}]")
    logging.info(f"  Original bounds: ({min_x:.2f}, {min_y:.2f}) - ({max_x:.2f}, {max_y:.2f})")
    logging.info(f"  Cropped bounds:  ({new_min_x:.2f}, {new_min_y:.2f}) - ({new_max_x:.2f}, {new_max_y:.2f})")

    return cropped, new_bounds


def save_image(
    image: np.ndarray,
    output_file: str,
    bounds: Tuple[float, float, float, float],
    settings: OrthomosaicSettings,
    dem_metadata: Optional['DEMMetadata'] = None
) -> None:
    """
    Save the orthomosaic image with optional georeferencing.

    Args:
        image: RGBA numpy array
        output_file: Output file path
        bounds: World coordinate bounds (local coordinates)
        settings: Orthomosaic settings
        dem_metadata: Optional DEM metadata (unused, kept for compatibility)
    """
    # Optionally crop to content (before applying coordinate offsets)
    if settings.crop_to_content:
        logging.info("Cropping to content...")
        image, bounds = crop_to_content(image, bounds, padding=0)

    output_lower = output_file.lower()
    height, width = image.shape[:2]

    # Apply coordinate offsets to convert from local to real-world coordinates
    min_x, min_y, max_x, max_y = bounds

    geo_bounds = (
        min_x + settings.coord_offset_x + settings.manual_offset_x,
        min_y + settings.coord_offset_y + settings.manual_offset_y,
        max_x + settings.coord_offset_x + settings.manual_offset_x,
        max_y + settings.coord_offset_y + settings.manual_offset_y
    )

    logging.info(f"Coordinate information:")
    logging.info(f"  Local bounds: ({min_x:.2f}, {min_y:.2f}) - ({max_x:.2f}, {max_y:.2f})")
    logging.info(f"  Image extent: {max_x - min_x:.2f} x {max_y - min_y:.2f} m")
    if settings.coord_offset_x != 0 or settings.coord_offset_y != 0:
        logging.info(f"  Origin offset: X+{settings.coord_offset_x:.2f}, Y+{settings.coord_offset_y:.2f}")
    if settings.manual_offset_x != 0 or settings.manual_offset_y != 0:
        logging.info(f"  Manual offset: X+{settings.manual_offset_x:.2f}, Y+{settings.manual_offset_y:.2f}")
    logging.info(f"  Final UTM bounds: ({geo_bounds[0]:.2f}, {geo_bounds[1]:.2f}) - ({geo_bounds[2]:.2f}, {geo_bounds[3]:.2f})")

    # Save world file
    save_world_file(output_file, geo_bounds, width, height)

    # Save PRJ file if CRS is specified
    if settings.crs is not None:
        save_prj_file(output_file, settings.crs)

    # Determine if we should save as GeoTIFF
    use_geotiff = (
        settings.geotiff and
        HAS_RASTERIO and
        (output_lower.endswith('.tif') or output_lower.endswith('.tiff'))
    )

    if use_geotiff:
        success = save_geotiff(
            image=image,
            output_file=output_file,
            bounds=geo_bounds,
            crs=settings.crs,
            compression=settings.geotiff_compression,
            create_overviews=settings.create_overviews,
            overview_levels=settings.overview_levels
        )

        if success:
            return
        else:
            logging.warning("GeoTIFF save failed, falling back to OpenCV")

    # Fallback to OpenCV
    logging.info(f"Saving image with OpenCV to {output_file}")
    result_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    if settings.output_format.lower() == 'jpg' or output_lower.endswith('.jpg'):
        cv2.imwrite(output_file, result_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
    else:
        cv2.imwrite(output_file, result_bgr)

    logging.info(f"Image saved. Use the .tfw world file to georeference in GIS software.")


# ============================================================================
# Main Functions
# ============================================================================

def render_orthomosaic(
        dem_file: str,
        poses_file: str,
        images_folder: str,
        output_file: str,
        settings: Optional[OrthomosaicSettings] = None,
        mask_file: Optional[str] = None,
        correction_matrix_file: Optional[str] = None,
        onefile_corrections_file: Optional[str] = None,
        flight_key: Optional[str] = None,
        dem_metadata_file: Optional[str] = None
) -> None:
    """
    Main function to generate an orthomosaic.

    Args:
        dem_file: Path to the DEM/mesh file (GLB/GLTF)
        poses_file: Path to the matched poses JSON file
        images_folder: Path to the folder containing input images
        output_file: Path for the output orthomosaic
        settings: Orthomosaic generation settings
        mask_file: Optional mask file for vignetting correction
        correction_matrix_file: Path to the default/global correction JSON file
        onefile_corrections_file: Path to the unified corrections file with frame-specific corrections
        flight_key: Flight key for looking up corrections (if using unified corrections file)
        dem_metadata_file: Path to DEM metadata JSON with origin/CRS info (e.g., 14_dem_mesh_r2.json)
    """
    if settings is None:
        settings = OrthomosaicSettings()

    logging.info("=" * 70)
    logging.info("Starting orthomosaic generation")
    logging.info("=" * 70)

    # Load DEM metadata for georeferencing
    dem_metadata = None
    if dem_metadata_file:
        dem_metadata = load_dem_metadata(dem_metadata_file)

        if dem_metadata:
            # Apply origin offset from metadata if not already set in settings
            if settings.coord_offset_x == 0.0 and settings.coord_offset_y == 0.0:
                settings.coord_offset_x = dem_metadata.origin_x
                settings.coord_offset_y = dem_metadata.origin_y
                logging.info(f"Using origin offset from DEM metadata: "
                           f"X={settings.coord_offset_x:.3f}, Y={settings.coord_offset_y:.3f}")

            # Use CRS from metadata if not specified in settings
            if settings.crs is None:
                epsg_code = dem_metadata.get_epsg_code()
                if epsg_code:
                    settings.crs = epsg_code
                    logging.info(f"Using CRS from DEM metadata: EPSG:{settings.crs}")

    # Log frame filter settings
    if settings.frame_filter:
        logging.info(f"Frame filter: {settings.frame_filter}")
    else:
        logging.info("Frame filter: none (using all frames)")

    # Load mesh data
    logging.info(f"Loading DEM from {dem_file}")
    mesh_data, texture_data = read_gltf(dem_file)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    mesh_aabb = get_aabb(mesh_data.vertices)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)

    # Log mesh bounds (LOCAL coordinates)
    logging.info(f"Mesh bounds (LOCAL):")
    logging.info(f"  X: [{float(mesh_aabb.p_min.x):.2f}, {float(mesh_aabb.p_max.x):.2f}]")
    logging.info(f"  Y: [{float(mesh_aabb.p_min.y):.2f}, {float(mesh_aabb.p_max.y):.2f}]")
    logging.info(f"  Z: [{float(mesh_aabb.p_min.z):.2f}, {float(mesh_aabb.p_max.z):.2f}]")

    # Create OpenGL context
    ctx = make_mgl_context()

    # Load mask if provided
    mask = None
    if mask_file and os.path.exists(mask_file):
        logging.info(f"Loading mask from {mask_file}")
        mask_img = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        mask = TextureData(CtxShot._cvt_img(mask_img))

    # Load poses
    logging.info(f"Loading poses from {poses_file}")
    with open(poses_file, 'r') as f:
        matched_poses = json.load(f)

    # Load corrections
    corrections_data = load_corrections_data(
        correction_matrix_file=correction_matrix_file,
        onefile_corrections_file=onefile_corrections_file,
        flight_key=flight_key
    )
    logging.info(f"Loaded {len(corrections_data.get('corrections', []))} frame-specific corrections")

    # Load all shots
    logging.info("Loading shots")
    shots = load_all_shots(
        images_folder=images_folder,
        matched_poses=matched_poses,
        ctx=ctx,
        corrections_data=corrections_data,
        frame_filter=settings.frame_filter,
        default_fovy=settings.default_fovy,
        flight_key=flight_key
    )
    logging.info(f"Loaded {len(shots)} shots")

    if len(shots) == 0:
        logging.error("No shots found!")
        release_all(ctx)
        return

    # Log shot position range (LOCAL coordinates)
    shot_positions = np.array([shot.camera.transform.position for shot in shots])
    logging.info(f"Shot positions (LOCAL):")
    logging.info(f"  X: [{shot_positions[:, 0].min():.2f}, {shot_positions[:, 0].max():.2f}]")
    logging.info(f"  Y: [{shot_positions[:, 1].min():.2f}, {shot_positions[:, 1].max():.2f}]")
    logging.info(f"  Z: [{shot_positions[:, 2].min():.2f}, {shot_positions[:, 2].max():.2f}]")

    # Compute global bounds
    logging.info("Computing global bounds")
    global_bounds = compute_global_bounds(shots, mesh_aabb)
    logging.info(f"Global bounds (LOCAL): x=[{global_bounds[0]:.2f}, {global_bounds[2]:.2f}], "
                 f"y=[{global_bounds[1]:.2f}, {global_bounds[3]:.2f}]")

    # Preview UTM bounds
    utm_preview = (
        global_bounds[0] + settings.coord_offset_x,
        global_bounds[1] + settings.coord_offset_y,
        global_bounds[2] + settings.coord_offset_x,
        global_bounds[3] + settings.coord_offset_y
    )
    logging.info(f"Expected UTM bounds: x=[{utm_preview[0]:.2f}, {utm_preview[2]:.2f}], "
                 f"y=[{utm_preview[1]:.2f}, {utm_preview[3]:.2f}]")

    # Compute output resolution
    global_resolution = compute_output_resolution(global_bounds, settings.ground_resolution)
    logging.info(f"Output resolution: {global_resolution.width} x {global_resolution.height}")

    # Check if tiling is needed
    needs_tiling = (
            global_resolution.width > settings.max_tile_size or
            global_resolution.height > settings.max_tile_size
    )

    if needs_tiling:
        logging.info("Using tiled rendering")
        render_tiled_orthomosaic(
            ctx, mesh_data, texture_data, shots, mask,
            global_bounds, global_resolution, settings, output_file
        )
    else:
        logging.info("Using single-pass rendering")
        render_single_orthomosaic(
            ctx, mesh_data, texture_data, shots, mask,
            global_bounds, global_resolution, settings, output_file
        )

    # Cleanup
    logging.info("Cleaning up")
    release_all(ctx, shots)
    logging.info("Orthomosaic generation complete")


def load_all_shots(
        images_folder: str,
        matched_poses: dict,
        ctx: Context,
        corrections_data: Dict[str, Any],
        frame_filter: Optional[FrameFilter] = None,
        default_fovy: float = 60.0,
        flight_key: Optional[str] = None
) -> List[CtxShot]:
    """
    Load all shots from the images folder with proper corrections.

    Args:
        images_folder: Path to folder containing images
        matched_poses: Dictionary with pose data
        ctx: ModernGL context
        corrections_data: Dictionary with correction data
        frame_filter: Optional filter for frame range
        default_fovy: Default FOV if not specified in poses
        flight_key: Optional flight key for filtering files
    """
    shots = []

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Filter by flight key if provided
    if flight_key is not None:
        image_files = [f for f in image_files if f.split("_")[0] == str(flight_key)]

    loaded_count = 0
    skipped_filter = 0
    skipped_error = 0

    for img_file in image_files:
        try:
            # Extract frame index from filename (format: flightkey_frameindex.ext)
            parts = img_file.split("_")
            if len(parts) >= 2:
                idx = int(parts[1].split(".")[0])
            else:
                idx = int(Path(img_file).stem)

            # Apply frame filter
            if frame_filter is not None and not frame_filter.is_valid(idx):
                skipped_filter += 1
                continue

            if idx >= len(matched_poses["images"]):
                logging.warning(f"Frame index {idx} out of range for poses")
                skipped_error += 1
                continue

            pose = matched_poses["images"][idx]

            # Camera position
            camera_position = Vector3(pose["location"])

            # Camera rotation
            camera_rotation = pose["rotation"]
            camera_rotation = [val % 360.0 for val in camera_rotation]

            if len(camera_rotation) == 3:
                eulers = [np.deg2rad(val) for val in camera_rotation]
                camera_rotation = quaternion_from_eulers(eulers, 'zyx')
            elif len(camera_rotation) == 4:
                camera_rotation = Quaternion(camera_rotation)
            else:
                logging.warning(f"Invalid rotation format for {img_file}")
                skipped_error += 1
                continue

            # FOV
            fov = pose.get("fovy", [default_fovy])
            if fov is None:
                fov = default_fovy
            elif isinstance(fov, list):
                fov = fov[0] if fov else default_fovy

            # Get frame-specific correction
            frame_correction_dict = get_frame_correction(corrections_data, idx)
            correction, _ = correction_dict_to_transform(frame_correction_dict)

            # Create shot
            shot = CtxShot(
                ctx,
                os.path.join(images_folder, img_file),
                camera_position,
                camera_rotation,
                fov,
                1,
                correction,
                lazy=True
            )
            shots.append(shot)
            loaded_count += 1

        except (ValueError, KeyError, IndexError) as e:
            logging.warning(f"Skipping {img_file}: {e}")
            skipped_error += 1
            continue

    logging.info(f"Shot loading complete: {loaded_count} loaded, "
                 f"{skipped_filter} filtered out, {skipped_error} errors")

    return shots


def render_single_orthomosaic(
        ctx: Context,
        mesh_data,
        texture_data,
        shots: List[CtxShot],
        mask: Optional[TextureData],
        global_bounds: Tuple[float, float, float, float],
        global_resolution: Resolution,
        settings: OrthomosaicSettings,
        output_file: str
) -> None:
    """Render orthomosaic in a single pass (for smaller outputs)."""

    # Create global camera
    mesh_aabb = get_aabb(mesh_data.vertices)
    global_camera = create_global_orthographic_camera(
        global_bounds, mesh_aabb, settings
    )

    # Create renderer
    renderer = Renderer(global_resolution, ctx, global_camera, mesh_data, texture_data)

    if settings.blend_mode == BlendMode.INTEGRAL:
        # Light field integral mode - average all overlapping pixels
        logging.info("Rendering with INTEGRAL blend mode (light field averaging)")
        shot_loader = make_shot_loader(shots)
        result = renderer.render_integral(
            shot_loader,
            mask=mask,
            save=False,
            release_shots=False,
            auto_contrast=settings.auto_contrast,
            alpha_threshold=settings.alpha_threshold
        )
    else:
        # Orthomosaic modes - process shots individually
        result = render_orthomosaic_sequential(
            renderer, shots, mask, global_resolution, settings, global_bounds
        )

    # Save result with georeferencing
    save_image(result, output_file, global_bounds, settings)
    renderer.release()


def render_orthomosaic_sequential(
        renderer: Renderer,
        shots: List[CtxShot],
        mask: Optional[TextureData],
        resolution: Resolution,
        settings: OrthomosaicSettings,
        global_bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Render orthomosaic by processing shots sequentially with first/last/center priority.

    This produces a "true" orthomosaic without light field blending - each pixel
    comes from exactly one source image.

    Args:
        renderer: The Renderer instance
        shots: List of shots to process
        mask: Optional vignetting mask
        resolution: Output resolution
        settings: Orthomosaic settings (contains blend_mode)
        global_bounds: World coordinate bounds for center priority calculation

    Returns:
        RGBA numpy array with the final orthomosaic
    """
    blend_mode = settings.blend_mode
    logging.info(f"Rendering with {blend_mode.value.upper()} blend mode (true orthomosaic)")

    # Initialize output array
    # Using float32 for accumulation, will convert to uint8 at the end
    output = np.zeros((resolution.height, resolution.width, 4), dtype=np.uint8)

    # Track which pixels have been filled (for FIRST mode)
    # Alpha channel serves this purpose - alpha > 0 means pixel is filled

    # For CENTER mode, we need to track distance to camera for each pixel
    if blend_mode == BlendMode.CENTER:
        # Store the squared distance from each pixel to the camera that filled it
        # Initialize with infinity
        distance_map = np.full((resolution.height, resolution.width), np.inf, dtype=np.float32)

        # Compute pixel coordinates in world space (center of output)
        min_x, min_y, max_x, max_y = global_bounds

    # Determine shot processing order
    if blend_mode == BlendMode.CENTER:
        # Sort shots by distance to center of bounds (process center shots last so they win)
        center_x = (global_bounds[0] + global_bounds[2]) / 2
        center_y = (global_bounds[1] + global_bounds[3]) / 2

        def distance_to_center(shot):
            pos = shot.camera.transform.position
            return (pos.x - center_x) ** 2 + (pos.y - center_y) ** 2

        shots_ordered = sorted(shots, key=distance_to_center, reverse=True)  # Farthest first
        logging.info("Shots sorted by distance to center (farthest first, closest wins)")
    else:
        shots_ordered = shots

    total_shots = len(shots_ordered)

    for i, shot in enumerate(shots_ordered):
        if (i + 1) % 10 == 0 or i == 0 or i == total_shots - 1:
            logging.info(f"Processing shot {i + 1}/{total_shots}")

        # Render this single shot
        shot_result = render_single_shot(renderer, shot, mask)

        if shot_result is None:
            continue

        # Get alpha mask for this shot (where shot has valid data)
        shot_alpha = shot_result[:, :, 3]
        shot_has_data = shot_alpha > 0

        if blend_mode == BlendMode.FIRST:
            # First-write-wins: only write where output is empty
            output_empty = output[:, :, 3] == 0
            write_mask = shot_has_data & output_empty

        elif blend_mode == BlendMode.LAST:
            # Last-write-wins: always overwrite
            write_mask = shot_has_data

        elif blend_mode == BlendMode.CENTER:
            # Center priority: write if this shot's camera is closer to the pixel
            # For simplicity, use distance from camera to pixel in XY plane

            # Get camera position
            cam_pos = shot.camera.transform.position
            cam_x, cam_y = float(cam_pos.x), float(cam_pos.y)

            # Create pixel coordinate grids
            min_x, min_y, max_x, max_y = global_bounds
            pixel_size_x = (max_x - min_x) / resolution.width
            pixel_size_y = (max_y - min_y) / resolution.height

            # Pixel centers in world coordinates
            px = np.arange(resolution.width) * pixel_size_x + min_x + pixel_size_x / 2
            py = np.arange(resolution.height) * pixel_size_y + min_y + pixel_size_y / 2
            px_grid, py_grid = np.meshgrid(px, py)

            # Distance from each pixel to this camera
            dist_sq = (px_grid - cam_x) ** 2 + (py_grid - cam_y) ** 2

            # Write where shot has data AND this camera is closer
            closer = dist_sq < distance_map
            write_mask = shot_has_data & closer

            # Update distance map where we wrote
            distance_map[write_mask] = dist_sq[write_mask]
        else:
            write_mask = shot_has_data

        # Apply the write mask
        for c in range(4):
            output[:, :, c][write_mask] = shot_result[:, :, c][write_mask]

    logging.info(f"Orthomosaic complete. Coverage: {np.mean(output[:, :, 3] > 0) * 100:.1f}%")

    return output


def render_single_shot(
        renderer: Renderer,
        shot: CtxShot,
        mask: Optional[TextureData]
) -> Optional[np.ndarray]:
    """
    Render a single shot and return the result as a numpy array.

    Args:
        renderer: The Renderer instance
        shot: The shot to render
        mask: Optional vignetting mask

    Returns:
        RGBA numpy array or None if rendering failed
    """
    try:
        # Use project_shots_iter to render single shot
        results = list(renderer.project_shots_iter(
            shot,
            RenderResultMode.ShotOnly,
            release_shots=False,
            mask=mask
        ))

        if results and len(results) > 0:
            return results[0]
        return None

    except Exception as e:
        logging.warning(f"Failed to render shot: {e}")
        return None


def render_tiled_orthomosaic(
        ctx: Context,
        mesh_data,
        texture_data,
        shots: List[CtxShot],
        mask: Optional[TextureData],
        global_bounds: Tuple[float, float, float, float],
        global_resolution: Resolution,
        settings: OrthomosaicSettings,
        output_file: str
) -> None:
    """Render orthomosaic using tiling (for large outputs)."""

    mesh_aabb = get_aabb(mesh_data.vertices)
    global_camera = create_global_orthographic_camera(
        global_bounds, mesh_aabb, settings
    )

    # Generate tiles
    tiles = generate_tiles(
        global_resolution,
        settings.max_tile_size,
        settings.tile_overlap
    )
    logging.info(f"Generated {len(tiles)} tiles")
    logging.info(f"Blend mode: {settings.blend_mode.value.upper()}")

    # Create output canvas (we'll assemble tiles into this)
    # For very large outputs, you might want to use memory-mapped arrays
    output = np.zeros(
        (global_resolution.height, global_resolution.width, 4),
        dtype=np.uint8
    )

    for i, tile in enumerate(tiles):
        x_start, y_start, tile_width, tile_height = tile
        logging.info(f"Rendering tile {i + 1}/{len(tiles)}: ({x_start}, {y_start}) {tile_width}x{tile_height}")

        # Create tile camera
        tile_camera = create_tile_camera(
            global_camera, global_bounds, global_resolution, tile
        )

        # Calculate tile bounds in world coordinates
        min_x, min_y, max_x, max_y = global_bounds
        pixel_size_x = (max_x - min_x) / global_resolution.width
        pixel_size_y = (max_y - min_y) / global_resolution.height

        tile_bounds = (
            min_x + x_start * pixel_size_x,
            max_y - (y_start + tile_height) * pixel_size_y,  # min_y of tile
            min_x + (x_start + tile_width) * pixel_size_x,
            max_y - y_start * pixel_size_y  # max_y of tile
        )

        # Create renderer for this tile
        tile_resolution = Resolution(tile_width, tile_height)
        renderer = Renderer(tile_resolution, ctx, tile_camera, mesh_data, texture_data)

        # Render tile based on blend mode
        if settings.blend_mode == BlendMode.INTEGRAL:
            shot_loader = make_shot_loader(shots)
            tile_result = renderer.render_integral(
                shot_loader,
                mask=mask,
                save=False,
                release_shots=False,
                auto_contrast=settings.auto_contrast,
                alpha_threshold=settings.alpha_threshold
            )
        else:
            tile_result = render_orthomosaic_sequential(
                renderer, shots, mask, tile_resolution, settings, tile_bounds
            )

        # Place tile in output (handle overlap with blending)
        # Simple approach: just overwrite (for non-overlapping or last-write-wins)
        y_end = min(y_start + tile_height, global_resolution.height)
        x_end = min(x_start + tile_width, global_resolution.width)

        output[y_start:y_end, x_start:x_end] = tile_result[:y_end - y_start, :x_end - x_start]

        renderer.release()

    # Save final output with georeferencing
    save_image(output, output_file, global_bounds, settings)


# ============================================================================
# CLI / Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Example usage
    FLIGHT_ID = "14"
    DEFAULT_DEM = rf"Z:\correction_data\{FLIGHT_ID}_dem.glb"
    DEFAULT_DEM_METADATA = rf"Z:\correction_data\{FLIGHT_ID}dem_mesh_r2.json"
    DEFAULT_POSES = rf"Z:\correction_data\{FLIGHT_ID}_matched_poses.json"
    DEFAULT_IMAGES = r"Z:\20250312_Dataset_trimmed\images\val"
    DEFAULT_OUTPUT = r"Z:\orthomosaic.tif"
    DEFAULT_CORRECTION = rf"Z:\correction_data\{FLIGHT_ID}_correction.json"
    DEFAULT_ONEFILE_CORRECTIONS = r"Z:\20250312_Dataset_trimmed\corrections.json"
    DEFAULT_MASK = rf"Z:\correction_data\{FLIGHT_ID}_mask_t.png"
    DEFAULT_FRAME_START = 0
    DEFAULT_FRAME_END = None
    DEFAULT_CRS = "EPSG:32632"
    DEFAULT_BLEND_MODE = "first"  # "integral", "first", "last", "center"
    CROP_TO_CONTENT = True

    # Get paths from environment or use defaults
    dem_file = os.environ.get("DEM_FILE", DEFAULT_DEM)
    dem_metadata_file = os.environ.get("DEM_METADATA_FILE", DEFAULT_DEM_METADATA)
    poses_file = os.environ.get("POSES_FILE", DEFAULT_POSES)
    images_folder = os.environ.get("IMAGES_FOLDER", DEFAULT_IMAGES)
    output_file = os.environ.get("OUTPUT_FILE", DEFAULT_OUTPUT)
    correction_file = os.environ.get("CORRECTION_FILE", DEFAULT_CORRECTION)
    onefile_corrections = os.environ.get("ONEFILE_CORRECTIONS", DEFAULT_ONEFILE_CORRECTIONS)
    mask_file = os.environ.get("MASK_FILE", DEFAULT_MASK)

    # Frame filter from environment
    frame_start = os.environ.get("FRAME_START", DEFAULT_FRAME_START)
    frame_end = os.environ.get("FRAME_END", DEFAULT_FRAME_END)
    frame_filter = None
    if frame_start is not None or frame_end is not None:
        frame_filter = FrameFilter(
            start=int(frame_start) if frame_start else None,
            end=int(frame_end) if frame_end else None
        )

    # CRS from environment (e.g., "32632" for UTM 32N, or "EPSG:31255" for Austria GK Central)
    crs_env = os.environ.get("CRS", DEFAULT_CRS)
    crs = None
    if crs_env:
        try:
            crs = int(crs_env)  # Try as EPSG code
        except ValueError:
            crs = crs_env  # Use as string (e.g., "EPSG:32632")

    # Blend mode from environment
    # Options: "integral" (default), "first", "last", "center"
    blend_mode_str = os.environ.get("BLEND_MODE", DEFAULT_BLEND_MODE).lower()
    blend_mode_map = {
        "integral": BlendMode.INTEGRAL,
        "first": BlendMode.FIRST,
        "last": BlendMode.LAST,
        "center": BlendMode.CENTER,
    }
    blend_mode = blend_mode_map.get(blend_mode_str, BlendMode.INTEGRAL)

    # Flight key (optional)
    flight_key = os.environ.get("FLIGHT_KEY", FLIGHT_ID)

    # Settings from environment
    settings = OrthomosaicSettings(
        ground_resolution=float(os.environ.get("GROUND_RESOLUTION", 0.05)),
        max_tile_size=int(os.environ.get("MAX_TILE_SIZE", 8192)),
        alpha_threshold=float(os.environ.get("ALPHA_THRESHOLD", 0.5)),
        frame_filter=frame_filter,
        blend_mode=blend_mode,
        crs=crs,
        geotiff=bool(int(os.environ.get("GEOTIFF", 1))),
        geotiff_compression=os.environ.get("GEOTIFF_COMPRESSION", "lzw"),
        create_overviews=bool(int(os.environ.get("CREATE_OVERVIEWS", 1))),
        coord_offset_x=float(os.environ.get("COORD_OFFSET_X", 0.0)),
        coord_offset_y=float(os.environ.get("COORD_OFFSET_Y", 0.0)),
        crop_to_content=bool(int(os.environ.get("CROP_TO_CONTENT", 1 if CROP_TO_CONTENT else 0))),
        manual_offset_x=float(os.environ.get("MANUAL_OFFSET_X", 0.0)),
        manual_offset_y=float(os.environ.get("MANUAL_OFFSET_Y", 0.0)),
    )

    render_orthomosaic(
        dem_file=dem_file,
        poses_file=poses_file,
        images_folder=images_folder,
        output_file=output_file,
        settings=settings,
        mask_file=mask_file,
        correction_matrix_file=correction_file,
        onefile_corrections_file=onefile_corrections,
        flight_key=flight_key,
        dem_metadata_file=dem_metadata_file
    )