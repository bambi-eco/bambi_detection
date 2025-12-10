#!/usr/bin/env python3
"""
Drone Video Frame to GeoTIFF Generator

This script projects drone video frames onto a Digital Elevation Model (DEM) and
exports the results as georeferenced GeoTIFF files.

Coordinate System: UTM 33N (EPSG:32633)

The coordinate flow:
1. DEM mesh vertices are in LOCAL coordinates (small values centered around origin)
2. Drone poses are in LOCAL coordinates (same system as mesh)
3. Correction offset aligns poses with mesh (applied to poses)
4. DEM origin offset converts LOCAL → UTM (applied only to final GeoTIFF bounds)

Usage:
    python drone_geotiff_generator.py --sequence-id <ID> --images-folder <path> \
        --data-folder <path> --output-folder <path> [options]

Requirements:
    pip install numpy rasterio scipy pillow pyrr trimesh opencv-python --break-system-packages
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds, Affine
    from rasterio.crs import CRS
except ImportError:
    print("rasterio not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio", "--break-system-packages"])
    import rasterio
    from rasterio.transform import from_bounds, Affine
    from rasterio.crs import CRS

try:
    from PIL import Image
except ImportError:
    print("Pillow not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "--break-system-packages"])
    from PIL import Image

try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import griddata
except ImportError:
    print("scipy not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--break-system-packages"])
    from scipy.signal import savgol_filter
    from scipy.interpolate import griddata

try:
    from pyrr import Quaternion
except ImportError:
    print("pyrr not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyrr", "--break-system-packages"])
    from pyrr import Quaternion

try:
    import trimesh
except ImportError:
    print("trimesh not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh", "--break-system-packages"])
    import trimesh

try:
    import cv2
except ImportError:
    print("opencv-python not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "--break-system-packages"])
    import cv2


@dataclass
class CameraParams:
    """Camera parameters for a single frame."""
    position: np.ndarray  # Local coordinates (same system as mesh)
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    fov_y: float  # Vertical field of view in degrees


class DroneGeoTIFFGenerator:
    """
    Generator for georeferenced GeoTIFF files from drone video frames.
    """

    def __init__(
            self,
            sequence_id: str,
            images_folder: str,
            data_folder: str,
            output_folder: str,
            apply_smoothing: bool = True,
            output_resolution: float = 0.1,  # meters per pixel
            crs: str = "EPSG:32633",  # UTM 33N
            mask_path: Optional[str] = None,
            subsample: int = 1,
            verbose: bool = True
    ):
        """
        Initialize the GeoTIFF generator.

        Args:
            sequence_id: Sequence identifier (e.g., "14_1")
            images_folder: Folder containing sequence images
            data_folder: Folder containing DEM, poses, and correction files
            output_folder: Folder for output GeoTIFF files
            apply_smoothing: Whether to apply pose smoothing
            output_resolution: Output GeoTIFF resolution in meters per pixel
            crs: Coordinate reference system (default: UTM 33N)
            mask_path: Path to binary mask for undistorted frames
            subsample: Mesh subsampling factor for faster processing
            verbose: Whether to print progress information
        """
        self.sequence_id = sequence_id
        self.images_folder = images_folder
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.output_resolution = output_resolution
        self.crs = CRS.from_string(crs)
        self.subsample = subsample
        self.verbose = verbose

        # Initialize data containers
        self.poses: Dict = {}
        self.translation: Dict = {'x': 0, 'y': 0, 'z': 0}
        self.rotation: Dict = {'x': 0, 'y': 0, 'z': 0}

        # DEM origin offset (converts local → UTM)
        self.x_offset: float = 0.0
        self.y_offset: float = 0.0
        self.z_offset: float = 0.0

        self.vertices: np.ndarray = np.array([])
        self.faces: np.ndarray = np.array([])

        # Mask for undistorted frames
        self.mask_path = mask_path
        self.mask_array: Optional[np.ndarray] = None

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        # Load all data
        self._load_data(apply_smoothing)
        self._load_mask()

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _load_data(self, apply_smoothing: bool):
        """Load all required data files."""
        self._log(f"Loading data for sequence: {self.sequence_id}")

        # Extract parent ID from sequence ID (e.g., "14" from "14_1")
        parent_id = self.sequence_id.split('_')[0]

        # Construct file paths
        dem_path = os.path.join(self.data_folder, f"{parent_id}_dem.glb")
        poses_path = os.path.join(self.data_folder, f"{parent_id}_matched_poses.json")
        correction_path = os.path.join(self.data_folder, f"{self.sequence_id}_correction.json")
        dem_meta_path = os.path.join(self.data_folder, f"{parent_id}_dem_mesh_r2.json")

        # Try alternative poses path if primary doesn't exist
        if not os.path.exists(poses_path):
            poses_path = os.path.join(self.data_folder, f"{self.sequence_id}_poses.json")

        # Load mesh (in local coordinates)
        self._log(f"Loading mesh from: {dem_path}")
        self.vertices, self.faces = self._load_glb_mesh(dem_path)
        self._log(f"  Loaded {len(self.vertices)} vertices and {len(self.faces)} faces")
        self._log(f"  Mesh bounds (local): X=[{self.vertices[:, 0].min():.2f}, {self.vertices[:, 0].max():.2f}]")
        self._log(f"                       Y=[{self.vertices[:, 1].min():.2f}, {self.vertices[:, 1].max():.2f}]")
        self._log(f"                       Z=[{self.vertices[:, 2].min():.2f}, {self.vertices[:, 2].max():.2f}]")

        # Load poses (in local coordinates)
        self._log(f"Loading poses from: {poses_path}")
        self.poses = self._load_poses(poses_path, apply_smoothing)
        self.n_frames = len(self.poses.get("images", []))
        self._log(f"  Loaded {self.n_frames} frames")

        # Load corrections (aligns poses with mesh in local coords)
        self._log(f"Loading corrections from: {correction_path}")
        self.translation, self.rotation = self._load_correction(correction_path)
        self._log(
            f"  Translation correction: ({self.translation['x']:.3f}, {self.translation['y']:.3f}, {self.translation['z']:.3f})")
        self._log(
            f"  Rotation correction: ({self.rotation['x']:.3f}, {self.rotation['y']:.3f}, {self.rotation['z']:.3f})")

        # Load DEM metadata (origin offset for local → UTM conversion)
        self._log(f"Loading DEM metadata from: {dem_meta_path}")
        self.x_offset, self.y_offset, self.z_offset = self._load_dem_metadata(dem_meta_path)
        self._log(f"  DEM origin (UTM offset): ({self.x_offset:.2f}, {self.y_offset:.2f}, {self.z_offset:.2f})")

        # Show what UTM bounds will be
        utm_x_min = self.vertices[:, 0].min() + self.x_offset
        utm_x_max = self.vertices[:, 0].max() + self.x_offset
        utm_y_min = self.vertices[:, 1].min() + self.y_offset
        utm_y_max = self.vertices[:, 1].max() + self.y_offset
        self._log(f"  Mesh bounds (UTM): X=[{utm_x_min:.2f}, {utm_x_max:.2f}]")
        self._log(f"                     Y=[{utm_y_min:.2f}, {utm_y_max:.2f}]")

    def _load_glb_mesh(self, glb_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load GLB mesh file."""
        if not os.path.exists(glb_path):
            raise FileNotFoundError(f"DEM file not found: {glb_path}")

        mesh = trimesh.load(glb_path, force='mesh')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        if self.subsample > 1:
            face_indices = np.arange(0, len(faces), self.subsample)
            faces = faces[face_indices]
            unique_vertex_indices = np.unique(faces.flatten())
            vertex_map = {old: new for new, old in enumerate(unique_vertex_indices)}
            vertices = vertices[unique_vertex_indices]
            faces = np.array([[vertex_map[v] for v in face] for face in faces])

        return vertices, faces

    def _load_poses(self, poses_path: str, apply_smoothing: bool) -> Dict:
        """Load poses from JSON."""
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"Poses file not found: {poses_path}")

        with open(poses_path, 'r') as f:
            poses = json.load(f)

        if apply_smoothing:
            images = poses.get("images", [])
            n = len(images)
            if n >= 5:
                window_length = min(11, n - 1 if (n - 1) % 2 == 1 else n - 2)
                if window_length >= 3:
                    positions = np.array([img["location"] for img in images], dtype=float)
                    smoothed = savgol_filter(positions, window_length=window_length,
                                             polyorder=2, axis=0, mode="interp")
                    for img, loc in zip(images, smoothed):
                        img["location"] = loc.tolist()

        return poses

    def _load_correction(self, correction_path: str) -> Tuple[Dict, Dict]:
        """Load correction data (aligns poses with mesh)."""
        if not os.path.exists(correction_path):
            self._log(f"  Warning: Correction file not found, using zero offsets")
            return {'x': 0, 'y': 0, 'z': 0}, {'x': 0, 'y': 0, 'z': 0}

        with open(correction_path, 'r') as f:
            correction = json.load(f)

        return (
            correction.get('translation', {'x': 0, 'y': 0, 'z': 0}),
            correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        )

    def _load_dem_metadata(self, dem_meta_path: str) -> Tuple[float, float, float]:
        """
        Load DEM metadata containing the origin offset.

        This offset converts local mesh coordinates to UTM coordinates:
        utm_x = local_x + origin[0]
        utm_y = local_y + origin[1]
        utm_z = local_z + origin[2]
        """
        if not os.path.exists(dem_meta_path):
            self._log(f"  Warning: DEM metadata file not found, using zero origin")
            return 0.0, 0.0, 0.0

        with open(dem_meta_path, 'r') as f:
            dem_meta = json.load(f)

        origin = dem_meta.get("origin", [0, 0, 0])
        return origin[0], origin[1], origin[2]

    def _load_mask(self):
        """Load binary mask for undistorted frames."""
        if self.mask_path is None:
            self._log("No mask path provided - transparency masking disabled")
            return

        if not os.path.exists(self.mask_path):
            self._log(f"Warning: Mask file not found: {self.mask_path}")
            return

        try:
            mask_img = Image.open(self.mask_path).convert('L')
            self.mask_array = np.array(mask_img)
            if self.mask_array.max() <= 1:
                self.mask_array = (self.mask_array * 255).astype(np.uint8)
            self._log(f"Loaded mask from: {self.mask_path}")
            self._log(f"  Mask shape: {self.mask_array.shape}")
        except Exception as e:
            self._log(f"Error loading mask: {e}")
            self.mask_array = None

    def _get_camera_params(self, frame_idx: int) -> CameraParams:
        """
        Get camera parameters for a frame with corrections applied.

        Camera position is in LOCAL coordinates (same as mesh).
        """
        pose_data = self.poses["images"][frame_idx]

        # Get raw position and apply correction offset (stays in local coords)
        position = np.array(pose_data['location'], dtype=float)
        position += np.array([
            self.translation['x'],
            self.translation['y'],
            self.translation['z']
        ])

        # Get rotation and apply correction
        rotation_deg = np.array(pose_data['rotation'], dtype=float)
        cor_rotation = np.array([
            self.rotation['x'],
            self.rotation['y'],
            self.rotation['z']
        ])

        # Apply rotation correction and convert to radians
        rotation_rad = np.deg2rad((rotation_deg % 360.0) - cor_rotation) * -1
        rotation = Quaternion.from_eulers(rotation_rad)
        rotation_matrix = np.array(rotation.matrix33)

        # Get field of view
        fovy = pose_data.get('fovy', 45.0)
        if isinstance(fovy, list):
            fovy = fovy[0]

        return CameraParams(position, rotation_matrix, fovy)

    def _find_image_path(self, frame_idx: int) -> Optional[str]:
        """Find image file for a frame."""
        images = self.poses.get("images", [])
        if frame_idx >= len(images):
            return None

        image_filename = images[frame_idx].get("imagefile", "")

        possible_paths = [
            os.path.join(self.images_folder, self.sequence_id, "img1", image_filename),
            os.path.join(self.images_folder, "img1", image_filename),
            os.path.join(self.images_folder, self.sequence_id, "img1", f"{frame_idx:08d}.png"),
            os.path.join(self.images_folder, self.sequence_id, "img1", f"{frame_idx:08d}.jpg"),
            os.path.join(self.images_folder, image_filename),
            os.path.join(self.images_folder, f"{frame_idx:08d}.png"),
            os.path.join(self.images_folder, f"{frame_idx:08d}.jpg"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _load_image(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load image for a frame."""
        image_path = self._find_image_path(frame_idx)
        if image_path is None:
            return None

        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            self._log(f"Error loading image: {e}")
            return None

    def _apply_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mask to image.

        Returns:
            Tuple of (image, mask) where mask is boolean array (True = valid pixel)
        """
        if self.mask_array is None:
            # No mask - all pixels valid
            return image, np.ones(image.shape[:2], dtype=bool)

        img_h, img_w = image.shape[:2]
        mask_h, mask_w = self.mask_array.shape

        # Resize mask if needed
        if (mask_h, mask_w) != (img_h, img_w):
            mask_img = Image.fromarray(self.mask_array)
            mask_img = mask_img.resize((img_w, img_h), Image.Resampling.NEAREST)
            mask = np.array(mask_img)
        else:
            mask = self.mask_array

        # Convert to boolean mask
        valid_mask = mask > 127

        return image, valid_mask

    def _project_image_to_mesh(
            self,
            image: np.ndarray,
            camera: CameraParams
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project image onto mesh as UV coordinates.

        This matches the original visualization script's projection logic.

        Returns:
            Tuple of (uvs, valid_mask) where:
            - uvs: (N, 2) array of UV coordinates for each vertex
            - valid_mask: (N,) boolean array indicating vertices within frustum
        """
        n_vertices = len(self.vertices)
        uvs = np.zeros((n_vertices, 2))
        valid_mask = np.zeros(n_vertices, dtype=bool)

        img_h, img_w = image.shape[:2]
        aspect_ratio = img_w / img_h

        # Build camera transformation matrix (in local coordinates)
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = camera.rotation_matrix
        cam_to_world[:3, 3] = camera.position
        world_to_cam = np.linalg.inv(cam_to_world)

        # Calculate FOV
        fov_y_rad = np.deg2rad(camera.fov_y)
        fov_x_rad = 2 * np.arctan(aspect_ratio * np.tan(fov_y_rad / 2))

        half_width = np.tan(fov_x_rad / 2)
        half_height = np.tan(fov_y_rad / 2)

        # Project each vertex to image coordinates
        for i, vertex in enumerate(self.vertices):
            vertex_h = np.append(vertex, 1.0)
            vertex_cam = world_to_cam @ vertex_h

            # Behind camera - skip
            if vertex_cam[2] >= 0:
                uvs[i] = [0.5, 0.5]
                continue

            # Project to normalized device coordinates
            x_ndc = vertex_cam[0] / (-vertex_cam[2])
            y_ndc = vertex_cam[1] / (-vertex_cam[2])

            # Convert to UV coordinates [0, 1]
            # Only V is flipped to correct vertical mirroring
            u = (x_ndc / half_width + 1) / 2
            v = 1.0 - (y_ndc / half_height + 1) / 2

            # Check if within valid UV range
            if 0 <= u <= 1 and 0 <= v <= 1:
                valid_mask[i] = True
                uvs[i] = [u, v]
            else:
                uvs[i] = [np.clip(u, 0, 1), np.clip(v, 0, 1)]

        return uvs, valid_mask

    def _create_geotiff_from_projection(
            self,
            image: np.ndarray,
            image_mask: np.ndarray,
            uvs: np.ndarray,
            valid_vertex_mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        """
        Create a georeferenced image from the projection.

        Args:
            image: Source image (H, W, C)
            image_mask: Boolean mask for valid image pixels
            uvs: UV coordinates for each mesh vertex
            valid_vertex_mask: Boolean mask for vertices within camera frustum

        Returns:
            Tuple of (output_image, output_mask, utm_bounds)
        """
        img_h, img_w = image.shape[:2]

        # Get valid vertices
        valid_indices = np.where(valid_vertex_mask)[0]
        if len(valid_indices) < 3:
            return None, None, None

        valid_vertices = self.vertices[valid_indices]
        valid_uvs = uvs[valid_indices]

        # Calculate bounds in LOCAL coordinates
        local_min_x = valid_vertices[:, 0].min()
        local_max_x = valid_vertices[:, 0].max()
        local_min_y = valid_vertices[:, 1].min()
        local_max_y = valid_vertices[:, 1].max()

        # Calculate output dimensions
        width_m = local_max_x - local_min_x
        height_m = local_max_y - local_min_y

        out_width = int(np.ceil(width_m / self.output_resolution))
        out_height = int(np.ceil(height_m / self.output_resolution))

        if out_width <= 0 or out_height <= 0:
            return None, None, None

        # Limit maximum dimensions
        max_dim = 8000
        if out_width > max_dim or out_height > max_dim:
            scale = max_dim / max(out_width, out_height)
            out_width = int(out_width * scale)
            out_height = int(out_height * scale)

        self._log(f"  Output size: {out_width}x{out_height} pixels")

        # Create output grid in LOCAL coordinates
        # X increases left to right, Y increases bottom to top (geographic convention)
        x_coords = np.linspace(local_min_x, local_max_x, out_width)
        y_coords = np.linspace(local_max_y, local_min_y, out_height)  # Top to bottom for image

        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Interpolate UV coordinates at grid points
        try:
            interp_u = griddata(
                valid_vertices[:, :2], valid_uvs[:, 0],
                grid_points, method='linear', fill_value=-1
            ).reshape(out_height, out_width)

            interp_v = griddata(
                valid_vertices[:, :2], valid_uvs[:, 1],
                grid_points, method='linear', fill_value=-1
            ).reshape(out_height, out_width)
        except Exception as e:
            self._log(f"  Interpolation error: {e}")
            return None, None, None

        # Convert UV to pixel coordinates
        # U: 0=left, 1=right -> pixel_x = u * (width - 1)
        # V: 0=top, 1=bottom -> pixel_y = v * (height - 1)
        pixel_x = interp_u * (img_w - 1)
        pixel_y = interp_v * (img_h - 1)

        # Valid output pixels (within image bounds and interpolation worked)
        valid_output = (
                (interp_u >= 0) & (interp_u <= 1) &
                (interp_v >= 0) & (interp_v <= 1)
        )

        # Create output arrays
        n_channels = image.shape[2] if len(image.shape) == 3 else 1
        if n_channels == 1:
            output_image = np.zeros((out_height, out_width), dtype=np.uint8)
        else:
            output_image = np.zeros((out_height, out_width, n_channels), dtype=np.uint8)

        output_valid = np.zeros((out_height, out_width), dtype=bool)

        # Sample pixels using bilinear interpolation
        valid_rows, valid_cols = np.where(valid_output)

        for row, col in zip(valid_rows, valid_cols):
            px = pixel_x[row, col]
            py = pixel_y[row, col]

            # Bilinear interpolation
            x0, y0 = int(px), int(py)
            x1, y1 = min(x0 + 1, img_w - 1), min(y0 + 1, img_h - 1)

            fx, fy = px - x0, py - y0

            # Check if source pixels are valid (in mask)
            if image_mask[y0, x0] and image_mask[y0, x1] and image_mask[y1, x0] and image_mask[y1, x1]:
                if n_channels == 1:
                    val = (
                            image[y0, x0] * (1 - fx) * (1 - fy) +
                            image[y0, x1] * fx * (1 - fy) +
                            image[y1, x0] * (1 - fx) * fy +
                            image[y1, x1] * fx * fy
                    )
                    output_image[row, col] = int(val)
                else:
                    for c in range(n_channels):
                        val = (
                                image[y0, x0, c] * (1 - fx) * (1 - fy) +
                                image[y0, x1, c] * fx * (1 - fy) +
                                image[y1, x0, c] * (1 - fx) * fy +
                                image[y1, x1, c] * fx * fy
                        )
                        output_image[row, col, c] = int(val)
                output_valid[row, col] = True

        # Convert bounds to UTM by adding offset
        utm_bounds = (
            local_min_x + self.x_offset,  # min_x (west)
            local_min_y + self.y_offset,  # min_y (south)
            local_max_x + self.x_offset,  # max_x (east)
            local_max_y + self.y_offset  # max_y (north)
        )

        return output_image, output_valid, utm_bounds

    def _save_geotiff(
            self,
            image: np.ndarray,
            valid_mask: np.ndarray,
            bounds: Tuple[float, float, float, float],
            output_path: str
    ):
        """
        Save image as a georeferenced GeoTIFF.

        Args:
            image: Image array (H, W) or (H, W, C)
            valid_mask: Boolean mask for valid pixels
            bounds: (min_x, min_y, max_x, max_y) in UTM coordinates
            output_path: Output file path
        """
        min_x, min_y, max_x, max_y = bounds
        height, width = image.shape[:2]

        # Create affine transform
        # GeoTIFF convention: origin at top-left, Y decreases downward
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        # Determine number of bands and prepare data
        if len(image.shape) == 2:
            count = 1
            data = image[np.newaxis, :, :]
        else:
            count = image.shape[2]
            # Rasterio expects (bands, height, width)
            data = np.moveaxis(image, -1, 0)

        # Set nodata value (0 for invalid pixels)
        nodata = 0

        # Apply mask - set invalid pixels to nodata
        for band in range(count):
            data[band][~valid_mask] = nodata

        # Write GeoTIFF
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=data.dtype,
                crs=self.crs,
                transform=transform,
                nodata=nodata,
                compress='lzw'
        ) as dst:
            dst.write(data)

        self._log(f"  Saved: {output_path}")
        self._log(f"  UTM bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")

    def process_frame(self, frame_idx: int) -> Optional[str]:
        """
        Process a single frame and generate GeoTIFF.

        Args:
            frame_idx: Frame index to process

        Returns:
            Output file path if successful, None otherwise
        """
        self._log(f"\nProcessing frame {frame_idx}/{self.n_frames - 1}")

        # Load image
        image = self._load_image(frame_idx)
        if image is None:
            self._log(f"  Warning: Could not load image for frame {frame_idx}")
            return None

        # Apply mask
        image, image_mask = self._apply_mask(image)

        # Get camera parameters (in local coordinates)
        camera = self._get_camera_params(frame_idx)
        self._log(
            f"  Camera position (local): ({camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f})")
        self._log(
            f"  Camera position (UTM): ({camera.position[0] + self.x_offset:.2f}, {camera.position[1] + self.y_offset:.2f}, {camera.position[2] + self.z_offset:.2f})")

        # Project image onto mesh
        uvs, valid_vertex_mask = self._project_image_to_mesh(image, camera)
        n_valid = np.sum(valid_vertex_mask)
        self._log(f"  Valid vertices in frustum: {n_valid}/{len(self.vertices)}")

        if n_valid < 10:
            self._log(f"  Warning: Too few valid vertices for frame {frame_idx}")
            return None

        # Create georeferenced output
        output_image, output_mask, utm_bounds = self._create_geotiff_from_projection(
            image, image_mask, uvs, valid_vertex_mask
        )

        if output_image is None:
            self._log(f"  Warning: Projection failed for frame {frame_idx}")
            return None

        # Generate output filename
        output_filename = f"{self.sequence_id}_frame_{frame_idx:06d}.tif"
        output_path = os.path.join(self.output_folder, output_filename)

        # Save GeoTIFF
        self._save_geotiff(output_image, output_mask, utm_bounds, output_path)

        return output_path

    def process_all_frames(
            self,
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            step: int = 1
    ) -> List[str]:
        """
        Process multiple frames and generate GeoTIFFs.

        Args:
            start_frame: First frame to process
            end_frame: Last frame to process (exclusive), None for all
            step: Process every Nth frame

        Returns:
            List of output file paths
        """
        if end_frame is None:
            end_frame = self.n_frames

        output_paths = []
        for frame_idx in range(start_frame, end_frame, step):
            path = self.process_frame(frame_idx)
            if path is not None:
                output_paths.append(path)

        self._log(f"\n{'=' * 60}")
        self._log(f"Processed {len(output_paths)} frames successfully")
        self._log(f"Output folder: {self.output_folder}")
        return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate georeferenced GeoTIFF files from drone video frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python drone_geotiff_generator.py \\
        --sequence-id 14_1 \\
        --images-folder /path/to/sequences/test \\
        --data-folder /path/to/correction_data \\
        --output-folder /path/to/geotiffs \\
        --resolution 0.1
        """
    )

    parser.add_argument('--sequence-id', '-s', type=str, default="14_1",
                        help='Sequence identifier (e.g., "14_1")')
    parser.add_argument('--images-folder', '-i', type=str, default=r"Z:\sequences\test",
                        help='Folder containing sequence images')
    parser.add_argument('--data-folder', '-d', type=str, default=r"Z:\correction_data",
                        help='Folder containing DEM, poses, and correction files')
    parser.add_argument('--output-folder', '-o', type=str, default=r"Z:\geotiffs",
                        help='Output folder for GeoTIFF files')

    parser.add_argument('--resolution', type=float, default=0.1,
                        help='Output resolution in meters per pixel (default: 0.1)')
    parser.add_argument('--crs', type=str, default='EPSG:32633',
                        help='Coordinate reference system (default: EPSG:32633 for UTM 33N)')

    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame to process (default: 0)')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Last frame to process (default: all)')
    parser.add_argument('--step', type=int, default=1,
                        help='Process every Nth frame (default: 1)')

    parser.add_argument('--no-smoothing', action='store_true',
                        help='Disable pose smoothing')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Mesh subsampling factor (default: 1)')

    parser.add_argument('--mask-path', '-m', type=str, default=None,
                        help='Path to binary mask image for undistorted frames')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.images_folder):
        print(f"Error: Images folder does not exist: {args.images_folder}")
        sys.exit(1)

    if not os.path.isdir(args.data_folder):
        print(f"Error: Data folder does not exist: {args.data_folder}")
        sys.exit(1)

    # Create generator
    generator = DroneGeoTIFFGenerator(
        sequence_id=args.sequence_id,
        images_folder=args.images_folder,
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        apply_smoothing=not args.no_smoothing,
        output_resolution=args.resolution,
        crs=args.crs,
        mask_path=args.mask_path,
        subsample=args.subsample,
        verbose=not args.quiet
    )

    # Process frames
    output_paths = generator.process_all_frames(
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        step=args.step
    )

    print(f"\nGenerated {len(output_paths)} GeoTIFF files in: {args.output_folder}")


if __name__ == "__main__":
    main()