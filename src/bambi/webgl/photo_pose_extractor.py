import glob
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pyproj
from pyproj import Transformer

import datetime
import os
import re

import cv2
import numpy as np
import numpy.typing as npt

from dateutil import tz

from bambi.airdata.air_data_parser import AirDataParser
from bambi.airdata.air_data_frame import AirDataFrame

# DJI filename pattern: DJI_YYYYMMDDHHMMSS_NNNN_X.JPG (or .jpg)
DJI_FILENAME_PATTERN = re.compile(
    r"DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_\d+.*\.\w+$"
)


def _parse_timestamp_from_filename(filename: str) -> Optional[datetime.datetime]:
    """
    Extract a naive datetime from a DJI-style filename.
    Returns None if the filename doesn't match the expected pattern.
    """
    basename = os.path.basename(filename)
    match = DJI_FILENAME_PATTERN.match(basename)
    if not match:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    return datetime.datetime(year, month, day, hour, minute, second)


def _parse_timestamp_from_exif(image_path: str) -> Optional[datetime.datetime]:
    """
    Extract a naive datetime from EXIF DateTimeOriginal (fallback: DateTime).
    Requires Pillow.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None

        # Build a tag-name → value lookup
        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

        for tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            value = exif.get(tag)
            if value:
                return datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def get_photo_timestamp(
    image_path: str, timezone_offset_hours: float
) -> Optional[datetime.datetime]:
    """
    Get a UTC-aware datetime for a drone photo.

    Tries the filename first, falls back to EXIF. The resulting naive
    local time is converted to UTC using the provided offset.

    :param image_path: Path to the image file.
    :param timezone_offset_hours: Offset from UTC of the photo timestamps
                                  (e.g. 1 for CET, 2 for CEST).
    :return: Timezone-aware datetime in UTC, or None on failure.
    """
    local_dt = _parse_timestamp_from_filename(image_path)
    if local_dt is None:
        local_dt = _parse_timestamp_from_exif(image_path)
    if local_dt is None:
        return None

    # Attach the local timezone and convert to UTC
    local_tz = tz.tzoffset(None, int(timezone_offset_hours * 3600))
    local_dt = local_dt.replace(tzinfo=local_tz)
    utc_dt = local_dt.astimezone(tz.tzutc())
    return utc_dt


def match_photos_to_airdata(
    image_paths: List[str],
    airdata_csv: str,
    photo_timezone_offset_hours: float = 1.0,
    max_time_delta_seconds: float = 10.0,
) -> Dict[str, Optional[AirDataFrame]]:
    """
    Match a list of drone photos to their AirData flight-log entries.

    For each image the function:
      1. Extracts a timestamp (filename → EXIF fallback).
      2. Converts it from local time to UTC.
      3. Finds the closest log frame that has ``isPhoto == 1``
         within *max_time_delta_seconds*.

    :param image_paths: List of paths to the drone images.
    :param airdata_csv: Path to the AirData CSV flight log.
    :param photo_timezone_offset_hours: UTC offset of the camera clock
        (e.g. 1 for CET, 2 for CEST, -5 for EST). Default: 1 (CET).
    :param max_time_delta_seconds: Maximum allowed time difference (in seconds)
        between a photo timestamp and a log entry. Photos without a match
        within this window map to ``None``.
    :return: Dict mapping each image *filename* (basename) to the best-matching
        :class:`AirDataFrame`, or ``None`` if no match was found.
    """
    # --- Parse the flight log once, keeping only isPhoto frames for matching ---
    parser = AirDataParser()
    all_frames = parser.parse(airdata_csv)
    photo_frames = [f for f in all_frames if f.isPhoto == 1]

    if not photo_frames:
        print(f"Warning: No isPhoto frames found in {airdata_csv}")
        return {os.path.basename(p): None for p in image_paths}

    max_delta = datetime.timedelta(seconds=max_time_delta_seconds)
    result: Dict[str, Optional[AirDataFrame]] = {}

    for image_path in image_paths:
        basename = os.path.basename(image_path)
        photo_utc = get_photo_timestamp(image_path, photo_timezone_offset_hours)

        if photo_utc is None:
            print(f"Warning: Could not extract timestamp from '{basename}'")
            result[basename] = None
            continue

        # Find the closest isPhoto frame by absolute time difference
        best_frame: Optional[AirDataFrame] = None
        best_diff = datetime.timedelta.max

        for frame in photo_frames:
            diff = abs(frame.datetime - photo_utc)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
            elif diff > best_diff:
                # Frames are chronological — no need to keep searching
                break

        if best_frame is not None and best_diff <= max_delta:
            result[basename] = best_frame
        else:
            print(
                f"Warning: No matching isPhoto frame for '{basename}' "
                f"(closest delta: {best_diff})"
            )
            result[basename] = None

    return result


class PhotoUndistorter:
    """
    Handles image undistortion for photos, replicating the logic of
    :class:`CalibratedVideoFrameAccessor`.

    The undistortion maps are lazily initialised on the first image so that
    the input resolution is detected automatically.
    """

    def __init__(
        self,
        calibration_res: Dict[str, Any],
        new_size: Optional[Tuple[int, int]] = None,
        new_camera_matrix: Optional[npt.NDArray[Any]] = None,
        alpha: float = 0.5,
        center_principal_point: bool = True,
        force_same_fov: bool = True,
    ):
        """
        :param calibration_res: Dict with ``"mtx"`` (3x3 camera matrix) and
            ``"dist"`` (distortion coefficients) -- same format used by
            :class:`CalibratedVideoFrameAccessor`.
        :param new_size: Target (width, height) after undistortion.  If
            ``None`` a square crop ``min(h, w) x min(h, w)`` is used (same
            default as the video accessor).
        :param new_camera_matrix: Explicit new camera matrix.  If ``None``
            it is computed via :func:`cv2.getOptimalNewCameraMatrix`.
        :param alpha: Free scaling parameter (0 = all valid pixels,
            1 = all source pixels retained).
        :param center_principal_point: Centre the principal point in the
            new matrix.
        :param force_same_fov: Enforce identical focal lengths in x and y
            (requires square ``new_size``).
        """
        self._mtx = np.asarray(calibration_res["mtx"])
        self._dist = np.asarray(calibration_res["dist"])
        self._requested_new_size = new_size
        self._requested_new_camera_matrix = (
            np.asarray(new_camera_matrix) if new_camera_matrix is not None else None
        )
        self._alpha = alpha
        self._center_pp = center_principal_point
        self._force_same_fov = force_same_fov

        # Lazily initialised on first image
        self._mapx: Optional[npt.NDArray[Any]] = None
        self._mapy: Optional[npt.NDArray[Any]] = None
        self._new_camera_matrix: Optional[npt.NDArray[Any]] = None
        self._new_size: Optional[Tuple[int, int]] = None
        self._is_initialized = False

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def fovy(self) -> float:
        """Vertical field of view in degrees, derived from the new camera
        matrix and the undistorted image height."""
        if not self._is_initialized:
            raise RuntimeError(
                "Undistortion has not been initialised yet -- call "
                "undistort() on at least one image first."
            )
        fy = self._new_camera_matrix[1, 1]
        h = self._new_size[1]
        return float(2.0 * math.degrees(math.atan(h / (2.0 * fy))))

    @property
    def new_camera_matrix(self) -> Optional[npt.NDArray[Any]]:
        return self._new_camera_matrix

    @property
    def new_size(self) -> Optional[Tuple[int, int]]:
        return self._new_size

    # ------------------------------------------------------------------
    # core methods
    # ------------------------------------------------------------------
    def prepare(self, img_size: Tuple[int, int]) -> None:
        """
        Compute undistortion maps for a given input image size.

        Mirrors :meth:`CalibratedVideoFrameAccessor.prepare_undistort`.

        :param img_size: ``(width, height)`` of the source images.
        """
        w, h = img_size

        # Determine target size
        if self._requested_new_size is not None:
            self._new_size = self._requested_new_size
        else:
            wh = min(h, w)
            self._new_size = (wh, wh)

        # Determine new camera matrix
        if self._requested_new_camera_matrix is not None:
            self._new_camera_matrix = self._requested_new_camera_matrix
        else:
            new_cameramtx, _roi = cv2.getOptimalNewCameraMatrix(
                self._mtx,
                self._dist,
                (w, h),
                self._alpha,
                self._new_size,
                centerPrincipalPoint=self._center_pp,
            )
            if self._force_same_fov:
                assert self._new_size[0] == self._new_size[1], (
                    "new_size must be square when force_same_fov is True!"
                )
                fxy = max(new_cameramtx[0, 0], new_cameramtx[1, 1])
                new_cameramtx[0, 0] = fxy
                new_cameramtx[1, 1] = fxy
            self._new_camera_matrix = new_cameramtx

        # Build remap LUTs
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self._mtx,
            self._dist,
            None,
            self._new_camera_matrix,
            self._new_size,
            cv2.CV_32FC1,
        )
        self._is_initialized = True

    def undistort(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Undistort a single image.

        On the first call the remap tables are initialised from the image
        dimensions (same lazy-init behaviour as the video accessor).

        :param img: BGR or grayscale input image.
        :return: Undistorted image.
        """
        if not self._is_initialized:
            h, w = img.shape[:2]
            self.prepare((w, h))

        return cv2.remap(img, self._mapx, self._mapy, cv2.INTER_LINEAR)

    def create_distortion_mask(
        self, width: int, height: int
    ) -> npt.NDArray[Any]:
        """Create a binary mask showing which pixels are valid after
        undistortion."""
        img = np.full((height, width), 255, dtype=np.uint8)
        return self.undistort(img)


class PhotoPoseExtractor:
    """
    Extracts camera poses for a set of drone photos and writes a
    ``poses.json`` compatible with the video-based :class:`PoseExtractor`.

    When *calibration_res* is provided the photos are undistorted using
    the same pipeline as :class:`CalibratedVideoFrameAccessor` and the
    vertical field-of-view is derived from the resulting camera matrix.
    """

    def __init__(
        self,
        rel_transformer: Transformer,
        fovy: Optional[float] = None,
        calibration_res: Optional[Dict[str, Any]] = None,
        new_size: Optional[Tuple[int, int]] = None,
        new_camera_matrix: Optional[npt.NDArray[Any]] = None,
        alpha: float = 0.5,
        center_principal_point: bool = True,
        force_same_fov: bool = True,
        apply_correction: bool = False,
        include_gps: bool = True,
        mask_images: bool = False,
    ):
        """
        :param rel_transformer: Transforms WGS-84 (lat, lon) to projected CRS.
        :param fovy: Manual vertical field-of-view in degrees.  Ignored when
            *calibration_res* is supplied (fovy is then computed from the
            new camera matrix).  Required when no calibration is given.
        :param calibration_res: Calibration dict with ``"mtx"`` and ``"dist"``
            keys.  When provided, images are undistorted and fovy is derived
            automatically.
        :param new_size: Target (width, height) for undistortion.
        :param new_camera_matrix: Explicit new camera matrix for undistortion.
        :param alpha: Free scaling parameter for
            :func:`cv2.getOptimalNewCameraMatrix` (0-1).
        :param center_principal_point: Centre the principal point.
        :param force_same_fov: Force equal fx/fy (requires square new_size).
        :param apply_correction: Apply grid-convergence heading correction.
        :param include_gps: Include ``lat``/``lng`` in per-image entries.
        :param mask_images: Write an alpha channel masking invalid (black)
            undistortion border pixels.
        """
        if calibration_res is None and fovy is None:
            raise ValueError(
                "Either 'fovy' or 'calibration_res' must be provided."
            )

        self.rel_transformer = rel_transformer
        self._manual_fovy = fovy
        self.apply_correction = apply_correction
        self.include_gps = include_gps
        self.mask_images = mask_images

        # Build undistorter if calibration is available
        self._undistorter: Optional[PhotoUndistorter] = None
        if calibration_res is not None:
            self._undistorter = PhotoUndistorter(
                calibration_res=calibration_res,
                new_size=new_size,
                new_camera_matrix=new_camera_matrix,
                alpha=alpha,
                center_principal_point=center_principal_point,
                force_same_fov=force_same_fov,
            )

    @property
    def fovy(self) -> float:
        """Effective vertical field of view (from calibration or manual)."""
        if self._undistorter is not None:
            return self._undistorter.fovy
        return self._manual_fovy

    @property
    def undistorter(self) -> Optional["PhotoUndistorter"]:
        return self._undistorter

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def extract(
        self,
        photo_dir: str,
        airdata_csv: str,
        output_path: str,
        output_image_dir: Optional[str] = None,
        photo_timezone_offset_hours: float = 1.0,
        max_time_delta_seconds: float = 10.0,
        origin: Optional[AirDataFrame] = None,
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None,
        origin_alt: Optional[float] = None,
        extensions: Tuple[str, ...] = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG",
                                        "*.tiff", "*.TIFF", "*.png", "*.PNG"),
    ) -> Dict[str, Any]:
        """
        Match photos to AirData entries, optionally undistort them, and
        write ``poses.json``.

        :param photo_dir: Directory containing the drone images.
        :param airdata_csv: Path to the AirData CSV flight log.
        :param output_path: Where to write the resulting ``poses.json``.
        :param output_image_dir: Directory for undistorted images.  Required
            when *calibration_res* was provided.  Ignored otherwise.
        :param photo_timezone_offset_hours: UTC offset of the camera clock.
        :param max_time_delta_seconds: Max seconds between photo and log entry.
        :param origin: Optional AirDataFrame used as coordinate origin.
        :param origin_lat: Latitude of the coordinate origin (WGS-84).
        :param origin_lon: Longitude of the coordinate origin (WGS-84).
        :param origin_alt: Altitude of the origin (m above sea level).
        :param extensions: Glob patterns for image files to include.
        :return: The poses dict (same object written to disk).
        """
        if self._undistorter is not None and output_image_dir is None:
            raise ValueError(
                "output_image_dir is required when calibration_res is "
                "provided (undistorted images need to be written somewhere)."
            )

        # -- collect image paths ------------------------------------------
        image_paths: List[str] = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(photo_dir, ext)))
        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise FileNotFoundError(
                f"No image files found in '{photo_dir}' "
                f"(extensions: {extensions})"
            )

        # Build a quick lookup from basename -> full source path
        basename_to_path = {os.path.basename(p): p for p in image_paths}

        # -- match photos to flight-log frames ----------------------------
        matches = match_photos_to_airdata(
            image_paths=image_paths,
            airdata_csv=airdata_csv,
            photo_timezone_offset_hours=photo_timezone_offset_hours,
            max_time_delta_seconds=max_time_delta_seconds,
        )

        # -- resolve origin -----------------------------------------------
        origin_frame = self._resolve_origin(
            origin, origin_lat, origin_lon, origin_alt, matches
        )
        origin_transformed = self.rel_transformer.transform(
            origin_frame.latitude, origin_frame.longitude
        )

        # -- prepare output image directory if undistorting ----------------
        if self._undistorter is not None:
            os.makedirs(output_image_dir, exist_ok=True)

        # -- build image entries sorted by timestamp ----------------------
        matched_items = [
            (filename, frame)
            for filename, frame in matches.items()
            if frame is not None
        ]
        matched_items.sort(key=lambda item: item[1].datetime)

        images: List[Dict[str, Any]] = []
        for filename, frame in matched_items:
            # Undistort and save if calibration is available
            if self._undistorter is not None:
                source_path = basename_to_path[filename]
                self._undistort_and_save(source_path, filename, output_image_dir)

            entry = self._build_image_entry(
                filename, frame, origin_frame, origin_transformed
            )
            images.append(entry)

        # -- assemble result dict (mirrors video PoseExtractor) -----------
        result: Dict[str, Any] = {
            "images": images,
            "origin": {
                "latitude": origin_frame.latitude,
                "longitude": origin_frame.longitude,
                "altitude": origin_frame.altitude,
            },
        }

        if self._undistorter is not None:
            result["mask"] = self._write_image_mask(output_image_dir)

        skipped = sum(1 for v in matches.values() if v is None)
        if skipped:
            print(
                f"Info: {skipped}/{len(matches)} photos could not be matched "
                f"and were skipped."
            )

        # -- write to disk ------------------------------------------------
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="UTF-8") as fh:
            json.dump(result, fh, indent=2)

        fovy_source = "calibration" if self._undistorter else "manual"
        print(
            f"Wrote {len(images)} photo poses to '{output_path}' "
            f"(origin: {origin_frame.latitude:.7f}, "
            f"{origin_frame.longitude:.7f} | "
            f"fovy: {self.fovy:.2f} deg [{fovy_source}])"
        )
        return result

    # ------------------------------------------------------------------
    # undistortion
    # ------------------------------------------------------------------
    def _write_image_mask(self, output_image_dir: str) -> str:
        """Write a binary undistortion mask to *output_image_dir* and return the filename."""
        filename = "mask.png"
        path = os.path.join(output_image_dir, filename)
        w, h = self._undistorter.new_size
        mask = self._undistorter.create_distortion_mask(w, h)
        mask[mask < 255] = 0
        cv2.imwrite(path, mask)
        return filename

    def _undistort_and_save(
        self, source_path: str, filename: str, output_dir: str
    ) -> str:
        """Load an image, undistort it, optionally apply alpha mask, and
        save to *output_dir*.  Returns the output path."""
        img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Could not read image: {source_path}")

        undistorted = self._undistorter.undistort(img)

        if self.mask_images:
            # Add alpha channel masking black border pixels
            # (same logic as PoseExtractorCallback)
            mask = np.full(
                (undistorted.shape[0], undistorted.shape[1]), 255, dtype=np.uint8
            )
            if undistorted.ndim == 3 and undistorted.shape[2] == 3:
                mask[np.all(undistorted == (0, 0, 0), axis=-1)] = 0
                undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2BGRA)
            elif undistorted.ndim == 2:
                mask[undistorted == 0] = 0
                undistorted = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGRA)
            undistorted[:, :, 3] = mask

        # Keep original filename but write as PNG (lossless, supports alpha)
        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, undistorted)
        return out_path

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_origin(
        origin: Optional[AirDataFrame],
        origin_lat: Optional[float],
        origin_lon: Optional[float],
        origin_alt: Optional[float],
        matches: Dict[str, Optional[AirDataFrame]],
    ) -> AirDataFrame:
        """Return an AirDataFrame that represents the coordinate origin."""
        if origin is not None:
            return origin

        if origin_lat is not None and origin_lon is not None:
            o = AirDataFrame()
            o.latitude = origin_lat
            o.longitude = origin_lon
            o.altitude = origin_alt if origin_alt is not None else 0.0
            return o

        first_frames = sorted(
            (f for f in matches.values() if f is not None),
            key=lambda f: f.datetime,
        )
        if not first_frames:
            raise ValueError(
                "No photos could be matched to the flight log -- "
                "cannot determine an origin automatically."
            )
        return first_frames[0]

    def _compute_correction_angle(self, frame: AirDataFrame) -> float:
        """Grid-convergence correction (same as video PoseExtractor)."""
        geod = pyproj.Geod(ellps="WGS84")
        target_crs = self.rel_transformer.target_crs
        cor_transformer = Transformer.from_crs("EPSG:4326", target_crs)

        frame_lon, frame_lat = frame.longitude, frame.latitude
        north_lon, north_lat, _ = geod.fwd(
            frame_lon, frame_lat, frame.gimbal_heading or 0.0, 1
        )
        north_lon_proj, north_lat_proj = cor_transformer.transform(
            north_lon, north_lat
        )
        lon_proj, lat_proj = self.rel_transformer.transform(frame_lat, frame_lon)

        return (
            90
            + math.atan2(
                lat_proj - north_lat_proj, lon_proj - north_lon_proj
            )
            * 180
            / math.pi
        )

    def _build_image_entry(
        self,
        filename: str,
        frame: AirDataFrame,
        origin_frame: AirDataFrame,
        origin_transformed: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Build a single image dict matching the video PoseExtractor schema."""
        frame_coord = self.rel_transformer.transform(
            frame.latitude, frame.longitude
        )
        frame_altitude = frame.altitude or 0.0
        origin_altitude = origin_frame.altitude or 0.0

        location = [
            frame_coord[0] - origin_transformed[0],
            frame_coord[1] - origin_transformed[1],
            frame_altitude - origin_altitude,
        ]

        correction_angle = (
            self._compute_correction_angle(frame)
            if self.apply_correction
            else 0.0
        )

        rotation = [
            float(frame.gimbal_pitch) + 90
            if frame.gimbal_pitch is not None
            else 0.0,
            0,  # roll is always zero
            (frame.compass_heading + correction_angle)
            if frame.compass_heading is not None
            else 0.0,
        ]

        # When undistorting, the output is always .png
        out_filename = filename
        if self._undistorter is not None:
            out_filename = os.path.splitext(filename)[0] + ".png"

        entry: Dict[str, Any] = {
            "imagefile": out_filename,
            "location": location,
            "rotation": rotation,
            "fovy": self.fovy,
            "timestamp": frame.datetime.isoformat(),
        }

        if self.include_gps:
            entry["lat"] = frame.latitude
            entry["lng"] = frame.longitude

        return entry