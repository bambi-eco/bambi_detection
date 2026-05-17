# -*- coding: utf-8 -*-
"""
Build AirData frames (and optionally a CSV) from DJI JPEG EXIF / XMP.

Used as a fallback when no AirData flight-log CSV is available and the
source images carry GPS / orientation metadata in their XMP segments.
Works for both thermal and RGB DJI JPEGs.

Unit / convention conversions applied here
------------------------------------------
* Altitude   : EXIF metres → stored as metres in AirDataFrame.
  AirDataWriter converts to feet when writing the CSV; AirDataParser
  converts back to metres on read — the round-trip is transparent.
* Gimbal pitch: no conversion — both EXIF and AirData use −90° = nadir.
* Gimbal roll : no conversion (0° = level in both conventions).
* Yaw angles  : EXIF −180…+180  →  AirData 0…360  (Python ``% 360``).
"""

import os
from typing import List

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_writer import AirDataWriter
from bambi.thermal.thermal_parser import read_dji_flight_params

__all__ = ['build_airdata_frames_from_exif', 'write_airdata_csv_from_exif']


def _airdata_frame_from_exif(params: dict, frame_id: int, time_ms: int) -> AirDataFrame:
    frame = AirDataFrame()
    frame.id = frame_id
    frame.time = time_ms
    frame.isPhoto = 1
    frame.isVideo = 0

    frame.datetime = params.get('datetime_utc')
    frame.latitude = params.get('latitude')
    frame.longitude = params.get('longitude')

    alt_abs = params.get('altitude_absolute')
    alt_rel = params.get('altitude_relative')
    frame.altitude = alt_abs
    frame.altitude_above_seaLevel = alt_abs
    frame.height_above_takeoff = alt_rel
    frame.ascent = alt_rel

    frame.gimbal_pitch = params.get('gimbal_pitch')
    gr = params.get('gimbal_roll')
    # At nadir, gimbal lock causes DJI to report ±180° roll instead of 0°
    frame.gimbal_roll = 0.0 if (gr is not None and abs(abs(gr) - 180.0) < 5.0) else gr
    gy = params.get('gimbal_yaw')
    frame.gimbal_heading = (gy % 360.0) if gy is not None else None

    # Flight orientation: normalise yaw to 0–360
    fy = params.get('flight_yaw')
    frame.compass_heading = (fy % 360.0) if fy is not None else None
    frame.pitch = params.get('flight_pitch')
    frame.roll = params.get('flight_roll')

    frame.xSpeed = params.get('flight_x_speed')
    frame.ySpeed = params.get('flight_y_speed')
    frame.zSpeed = params.get('flight_z_speed')

    return frame


def build_airdata_frames_from_exif(image_paths: List[str]) -> List[AirDataFrame]:
    """Create one :class:`AirDataFrame` per image from DJI XMP / EXIF.

    :param image_paths: Ordered list of image file paths.
    :returns: Frames in the same order as *image_paths*.  Images without GPS
        data produce a warning and are skipped.
    :raises ValueError: If no GPS data could be read from any image.
    """
    frames: List[AirDataFrame] = []
    failed: List[str] = []

    for i, path in enumerate(image_paths):
        params = read_dji_flight_params(path)
        if not params or 'latitude' not in params:
            failed.append(os.path.basename(path))
            print(f"Warning: No usable GPS/EXIF in '{os.path.basename(path)}'")
            continue
        frames.append(_airdata_frame_from_exif(params, frame_id=i, time_ms=i * 1000))

    if not frames:
        snippet = ', '.join(failed[:5]) + (' …' if len(failed) > 5 else '')
        raise ValueError(
            f"No GPS/EXIF data found in any of the {len(failed)} image(s). "
            f"First failed: {snippet}"
        )
    if failed:
        print(
            f"Warning: {len(failed)} image(s) skipped (no GPS/EXIF). "
            f"First few: {', '.join(failed[:3])}"
        )
    return frames


def write_airdata_csv_from_exif(image_paths: List[str], output_csv: str) -> int:
    """Write a synthetic AirData CSV reconstructed from image EXIF / XMP.

    :param image_paths: Ordered list of image file paths.
    :param output_csv: Destination path for the generated CSV.
    :returns: Number of rows written.
    :raises ValueError: If no GPS data could be read from any image.
    """
    frames = build_airdata_frames_from_exif(image_paths)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    AirDataWriter().write(output_csv, frames)
    return len(frames)
