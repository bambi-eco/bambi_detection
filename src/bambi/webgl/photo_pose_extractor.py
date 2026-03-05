import glob
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pyproj
from pyproj import Transformer

import datetime
import os
import re

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


class PhotoPoseExtractor:
    """
    Extracts camera poses for a set of drone photos and writes a
    ``poses.json`` compatible with the video-based :class:`PoseExtractor`.
    """

    def __init__(
        self,
        rel_transformer: Transformer,
        fovy: float,
        apply_correction: bool = False,
        include_gps: bool = True,
    ):
        """
        :param rel_transformer: Transforms WGS-84 (lat, lon) → projected CRS
            used for computing relative positions (e.g. EPSG:4326 → EPSG:32633).
        :param fovy: Vertical field-of-view of the camera in degrees.
        :param apply_correction: Apply grid-convergence correction to the
            compass heading (same logic as the video PoseExtractor).
        :param include_gps: Include ``lat`` / ``lng`` keys per image entry.
        """
        self.rel_transformer = rel_transformer
        self.fovy = fovy
        self.apply_correction = apply_correction
        self.include_gps = include_gps

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def extract(
        self,
        photo_dir: str,
        airdata_csv: str,
        output_path: str,
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
        Match photos to AirData entries and write ``poses.json``.

        The origin for the relative coordinate system can be supplied in
        three ways (checked in this order):

        1. An explicit :class:`AirDataFrame` via *origin*.
        2. Explicit *origin_lat* / *origin_lon* (/ *origin_alt*) values.
        3. ``None`` — the first matched photo's log entry is used.

        :param photo_dir: Directory containing the drone images.
        :param airdata_csv: Path to the AirData CSV flight log.
        :param output_path: Where to write the resulting ``poses.json``.
        :param photo_timezone_offset_hours: UTC offset of the camera clock.
        :param max_time_delta_seconds: Max seconds between photo timestamp
            and a log entry for a valid match.
        :param origin: Optional AirDataFrame used as coordinate origin.
        :param origin_lat: Latitude of the coordinate origin (WGS-84).
        :param origin_lon: Longitude of the coordinate origin (WGS-84).
        :param origin_alt: Altitude of the coordinate origin (m, above sea
            level). Defaults to 0 if not given.
        :param extensions: Glob patterns for image files to include.
        :return: The poses dict (same object that is written to disk).
        """
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

        # -- build image entries sorted by timestamp ----------------------
        matched_items = [
            (name, frame)
            for name, frame in matches.items()
            if frame is not None
        ]
        matched_items.sort(key=lambda item: item[1].datetime)

        images: List[Dict[str, Any]] = []
        for filename, frame in matched_items:
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

        print(
            f"Wrote {len(images)} photo poses to '{output_path}' "
            f"(origin: {origin_frame.latitude:.7f}, "
            f"{origin_frame.longitude:.7f})"
        )
        return result

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

        # fall back to the first matched photo (by timestamp)
        first_frames = sorted(
            (f for f in matches.values() if f is not None),
            key=lambda f: f.datetime,
        )
        if not first_frames:
            raise ValueError(
                "No photos could be matched to the flight log — "
                "cannot determine an origin automatically."
            )
        return first_frames[0]

    def _compute_correction_angle(
        self, frame: AirDataFrame
    ) -> float:
        """
        Compute the grid-convergence correction angle between geographic
        north and grid north at the frame's position (same method as the
        video PoseExtractor).
        """
        geod = pyproj.Geod(ellps="WGS84")
        target_crs = self.rel_transformer.target_crs
        cor_transformer = Transformer.from_crs("EPSG:4326", target_crs)

        frame_lon = frame.longitude
        frame_lat = frame.latitude

        # project a point 1 m to geographic north
        north_lon, north_lat, _ = geod.fwd(
            frame_lon, frame_lat, frame.gimbal_heading or 0.0, 1
        )
        north_lon_proj, north_lat_proj = cor_transformer.transform(
            north_lon, north_lat
        )

        lon_proj, lat_proj = self.rel_transformer.transform(frame_lat, frame_lon)

        correction_angle = (
            90
            + math.atan2(
                lat_proj - north_lat_proj,
                lon_proj - north_lon_proj,
            )
            * 180
            / math.pi
        )
        return correction_angle

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

        entry: Dict[str, Any] = {
            "imagefile": filename,
            "location": location,
            "rotation": rotation,
            "fovy": self.fovy,
            "timestamp": frame.datetime.isoformat(),
        }

        if self.include_gps:
            entry["lat"] = frame.latitude
            entry["lng"] = frame.longitude

        return entry