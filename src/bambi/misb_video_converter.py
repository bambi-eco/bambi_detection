#!/usr/bin/env python3
"""
MISB ST 0601 Video Converter

This script creates an MP4 video with embedded KLV metadata track from drone
video frames and pose data. The output is compatible with QGIS and other
GIS applications that support MISB ST 0601 metadata.

The script generates KLV (Key-Length-Value) metadata packets per MISB ST 0601.17
standard, which includes platform position, orientation, sensor parameters,
and timestamps.

Coordinate System: WGS84 (EPSG:4326) for lat/lon

Usage:
    python misb_video_converter.py --sequence-id <ID> --images-folder <path> \
        --poses-file <path> --output <path.mp4> [options]

Requirements:
    pip install numpy opencv-python --break-system-packages
    FFmpeg must be installed and available in PATH

References:
    - MISB ST 0601.17: UAS Datalink Local Set
    - MISB ST 0107.3: KLV encoding
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, BinaryIO
import shutil

try:
    import numpy as np
except ImportError:
    print("numpy not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--break-system-packages"])
    import numpy as np

try:
    import cv2
except ImportError:
    print("opencv-python not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "--break-system-packages"])
    import cv2

# MISB ST 0601 Universal Label (16 bytes)
# 06.0E.2B.34.02.0B.01.01.0E.01.03.01.01.00.00.00
MISB_UAS_LOCAL_SET_UL = bytes([
    0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
    0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00
])


# MISB ST 0601 Tag IDs
class MISBTag:
    CHECKSUM = 1
    PRECISION_TIME_STAMP = 2
    MISSION_ID = 3
    PLATFORM_TAIL_NUMBER = 4
    PLATFORM_HEADING = 5
    PLATFORM_PITCH = 6
    PLATFORM_ROLL = 7
    PLATFORM_TRUE_AIRSPEED = 8
    PLATFORM_INDICATED_AIRSPEED = 9
    PLATFORM_DESIGNATION = 10
    IMAGE_SOURCE_SENSOR = 11
    IMAGE_COORDINATE_SYSTEM = 12
    SENSOR_LATITUDE = 13
    SENSOR_LONGITUDE = 14
    SENSOR_TRUE_ALTITUDE = 15
    SENSOR_HORIZONTAL_FOV = 16
    SENSOR_VERTICAL_FOV = 17
    SENSOR_REL_AZIMUTH = 18
    SENSOR_REL_ELEVATION = 19
    SENSOR_REL_ROLL = 20
    SLANT_RANGE = 21
    TARGET_WIDTH = 22
    FRAME_CENTER_LATITUDE = 23
    FRAME_CENTER_LONGITUDE = 24
    FRAME_CENTER_ELEVATION = 25
    OFFSET_CORNER_LAT_1 = 26
    OFFSET_CORNER_LON_1 = 27
    OFFSET_CORNER_LAT_2 = 28
    OFFSET_CORNER_LON_2 = 29
    OFFSET_CORNER_LAT_3 = 30
    OFFSET_CORNER_LON_3 = 31
    OFFSET_CORNER_LAT_4 = 32
    OFFSET_CORNER_LON_4 = 33
    ICING_DETECTED = 34
    WIND_DIRECTION = 35
    WIND_SPEED = 36
    STATIC_PRESSURE = 37
    DENSITY_ALTITUDE = 38
    OUTSIDE_AIR_TEMP = 39
    TARGET_LOCATION_LAT = 40
    TARGET_LOCATION_LON = 41
    TARGET_LOCATION_ELEV = 42
    TARGET_TRK_GATE_WIDTH = 43
    TARGET_TRK_GATE_HEIGHT = 44
    TARGET_ERROR_EST_CE90 = 45
    TARGET_ERROR_EST_LE90 = 46
    GENERIC_FLAG_DATA = 47
    SECURITY_LOCAL_SET = 48
    DIFFERENTIAL_PRESSURE = 49
    PLATFORM_ANGLE_OF_ATTACK = 50
    PLATFORM_VERTICAL_SPEED = 51
    PLATFORM_SIDESLIP_ANGLE = 52
    AIRFIELD_BARO_PRESSURE = 53
    AIRFIELD_ELEVATION = 54
    RELATIVE_HUMIDITY = 55
    PLATFORM_GROUND_SPEED = 56
    GROUND_RANGE = 57
    PLATFORM_FUEL_REMAINING = 58
    PLATFORM_CALL_SIGN = 59
    WEAPON_LOAD = 60
    WEAPON_FIRED = 61
    LASER_PRF_CODE = 62
    SENSOR_FOV_NAME = 63
    PLATFORM_MAG_HEADING = 64
    UAS_LDS_VERSION = 65


@dataclass
class FramePose:
    """Pose data for a single frame."""
    timestamp: datetime
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters (MSL)
    heading: float  # degrees (0-360, north=0, clockwise)
    pitch: float  # degrees (-90 to 90)
    roll: float  # degrees (-180 to 180)
    fov_horizontal: float  # degrees
    fov_vertical: float  # degrees
    imagefile: str


class KLVEncoder:
    """
    Encoder for MISB ST 0601 KLV packets.

    Implements BER-OID length encoding and value encoding per MISB standards.
    """

    @staticmethod
    def encode_ber_length(length: int) -> bytes:
        """
        Encode length using BER (Basic Encoding Rules).

        Short form: length < 128 -> single byte
        Long form: length >= 128 -> 0x80 | num_bytes, followed by length bytes
        """
        if length < 128:
            return bytes([length])
        else:
            length_bytes = []
            temp = length
            while temp > 0:
                length_bytes.insert(0, temp & 0xFF)
                temp >>= 8
            return bytes([0x80 | len(length_bytes)] + length_bytes)

    @staticmethod
    def encode_ber_oid_length(length: int) -> bytes:
        """
        Encode length using BER-OID encoding (for tag values).

        Values 0-127: single byte
        Values >= 128: high bit set on continuation bytes
        """
        if length < 128:
            return bytes([length])

        result = []
        temp = length
        result.insert(0, temp & 0x7F)
        temp >>= 7

        while temp > 0:
            result.insert(0, (temp & 0x7F) | 0x80)
            temp >>= 7

        return bytes(result)

    @staticmethod
    def encode_uint8(value: int) -> bytes:
        """Encode unsigned 8-bit integer."""
        return struct.pack('>B', max(0, min(255, int(value))))

    @staticmethod
    def encode_uint16(value: int) -> bytes:
        """Encode unsigned 16-bit integer."""
        return struct.pack('>H', max(0, min(65535, int(value))))

    @staticmethod
    def encode_uint32(value: int) -> bytes:
        """Encode unsigned 32-bit integer."""
        return struct.pack('>I', max(0, min(4294967295, int(value))))

    @staticmethod
    def encode_uint64(value: int) -> bytes:
        """Encode unsigned 64-bit integer."""
        return struct.pack('>Q', max(0, int(value)))

    @staticmethod
    def encode_int16(value: int) -> bytes:
        """Encode signed 16-bit integer."""
        return struct.pack('>h', max(-32768, min(32767, int(value))))

    @staticmethod
    def encode_int32(value: int) -> bytes:
        """Encode signed 32-bit integer."""
        return struct.pack('>i', max(-2147483648, min(2147483647, int(value))))

    @staticmethod
    def encode_string(value: str) -> bytes:
        """Encode variable-length string (UTF-8)."""
        return value.encode('utf-8')

    @staticmethod
    def encode_latitude(lat: float) -> bytes:
        """
        Encode latitude per MISB ST 0601.

        Range: -90 to +90 degrees
        Encoding: signed 4-byte integer
        LSB: 90 / (2^31 - 1) degrees ≈ 4.19e-8 degrees
        """
        # Clamp to valid range
        lat = max(-90.0, min(90.0, lat))
        # Scale to 32-bit signed integer
        scaled = int(lat * (2 ** 31 - 1) / 90.0)
        return struct.pack('>i', scaled)

    @staticmethod
    def encode_longitude(lon: float) -> bytes:
        """
        Encode longitude per MISB ST 0601.

        Range: -180 to +180 degrees
        Encoding: signed 4-byte integer
        LSB: 180 / (2^31 - 1) degrees ≈ 8.38e-8 degrees
        """
        # Normalize to -180 to +180
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        # Scale to 32-bit signed integer
        scaled = int(lon * (2 ** 31 - 1) / 180.0)
        return struct.pack('>i', scaled)

    @staticmethod
    def encode_altitude(alt: float) -> bytes:
        """
        Encode altitude per MISB ST 0601.

        Range: -900 to +19000 meters
        Encoding: unsigned 2-byte integer
        LSB: (19000 - (-900)) / (2^16 - 1) ≈ 0.3 meters
        """
        # Clamp to valid range
        alt = max(-900.0, min(19000.0, alt))
        # Scale to 16-bit unsigned integer
        scaled = int((alt + 900.0) * (2 ** 16 - 1) / 19900.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_heading(heading: float) -> bytes:
        """
        Encode heading/azimuth per MISB ST 0601.

        Range: 0 to 360 degrees
        Encoding: unsigned 2-byte integer
        LSB: 360 / (2^16 - 1) degrees ≈ 0.0055 degrees
        """
        # Normalize to 0-360
        heading = heading % 360.0
        # Scale to 16-bit unsigned integer
        scaled = int(heading * (2 ** 16 - 1) / 360.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_platform_pitch(pitch: float) -> bytes:
        """
        Encode platform pitch per MISB ST 0601.

        Range: -20 to +20 degrees (platform pitch has limited range)
        Encoding: signed 2-byte integer
        LSB: 40 / (2^16 - 1) degrees
        """
        # Clamp to valid range
        pitch = max(-20.0, min(20.0, pitch))
        # Scale to 16-bit signed integer (offset by midpoint)
        scaled = int((pitch + 20.0) * (2 ** 16 - 1) / 40.0) - 32768
        return struct.pack('>h', scaled)

    @staticmethod
    def encode_platform_roll(roll: float) -> bytes:
        """
        Encode platform roll per MISB ST 0601.

        Range: -50 to +50 degrees
        Encoding: signed 2-byte integer
        LSB: 100 / (2^16 - 1) degrees
        """
        # Clamp to valid range
        roll = max(-50.0, min(50.0, roll))
        # Scale to 16-bit signed integer (offset by midpoint)
        scaled = int((roll + 50.0) * (2 ** 16 - 1) / 100.0) - 32768
        return struct.pack('>h', scaled)

    @staticmethod
    def encode_sensor_rel_angle(angle: float) -> bytes:
        """
        Encode sensor relative angle (azimuth/elevation/roll).

        Range: 0 to 360 degrees (for azimuth) or mapped appropriately
        Encoding: unsigned 4-byte integer
        """
        # Normalize to 0-360
        angle = angle % 360.0
        # Scale to 32-bit unsigned integer
        scaled = int(angle * (2 ** 32 - 1) / 360.0)
        return struct.pack('>I', scaled)

    @staticmethod
    def encode_fov(fov: float) -> bytes:
        """
        Encode field of view per MISB ST 0601.

        Range: 0 to 180 degrees
        Encoding: unsigned 2-byte integer
        LSB: 180 / (2^16 - 1) degrees
        """
        # Clamp to valid range
        fov = max(0.0, min(180.0, fov))
        # Scale to 16-bit unsigned integer
        scaled = int(fov * (2 ** 16 - 1) / 180.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_timestamp(dt: datetime) -> bytes:
        """
        Encode UNIX timestamp in microseconds since 1970-01-01 00:00:00 UTC.

        Per MISB ST 0601 Tag 2 (Precision Time Stamp).
        """
        # Convert to UTC if timezone-aware
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)

        # Get microseconds since epoch
        timestamp_us = int(dt.timestamp() * 1_000_000)
        return struct.pack('>Q', timestamp_us)

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        """
        Calculate 16-bit checksum per MISB ST 0601.

        Running 16-bit sum of all bytes in the Local Set,
        including the 16-byte UL key and all value bytes.
        """
        checksum = 0
        for i, byte in enumerate(data):
            checksum += byte << (8 * ((i + 1) % 2))
        return checksum & 0xFFFF


class MISBPacketBuilder:
    """
    Builder for MISB ST 0601 UAS Local Set packets.
    """

    def __init__(self):
        self.items: List[Tuple[int, bytes]] = []
        self.encoder = KLVEncoder()

    def add_item(self, tag: int, value: bytes):
        """Add a tag-value pair to the packet."""
        self.items.append((tag, value))

    def add_timestamp(self, dt: datetime):
        """Add precision timestamp (Tag 2)."""
        self.add_item(MISBTag.PRECISION_TIME_STAMP,
                      self.encoder.encode_timestamp(dt))

    def add_mission_id(self, mission_id: str):
        """Add mission ID (Tag 3)."""
        self.add_item(MISBTag.MISSION_ID,
                      self.encoder.encode_string(mission_id))

    def add_platform_tail_number(self, tail_number: str):
        """Add platform tail number (Tag 4)."""
        self.add_item(MISBTag.PLATFORM_TAIL_NUMBER,
                      self.encoder.encode_string(tail_number))

    def add_platform_heading(self, heading: float):
        """Add platform heading angle (Tag 5)."""
        self.add_item(MISBTag.PLATFORM_HEADING,
                      self.encoder.encode_heading(heading))

    def add_platform_pitch(self, pitch: float):
        """Add platform pitch angle (Tag 6)."""
        self.add_item(MISBTag.PLATFORM_PITCH,
                      self.encoder.encode_platform_pitch(pitch))

    def add_platform_roll(self, roll: float):
        """Add platform roll angle (Tag 7)."""
        self.add_item(MISBTag.PLATFORM_ROLL,
                      self.encoder.encode_platform_roll(roll))

    def add_platform_designation(self, designation: str):
        """Add platform designation (Tag 10)."""
        self.add_item(MISBTag.PLATFORM_DESIGNATION,
                      self.encoder.encode_string(designation))

    def add_image_source_sensor(self, sensor: str):
        """Add image source sensor (Tag 11)."""
        self.add_item(MISBTag.IMAGE_SOURCE_SENSOR,
                      self.encoder.encode_string(sensor))

    def add_sensor_latitude(self, lat: float):
        """Add sensor latitude (Tag 13)."""
        self.add_item(MISBTag.SENSOR_LATITUDE,
                      self.encoder.encode_latitude(lat))

    def add_sensor_longitude(self, lon: float):
        """Add sensor longitude (Tag 14)."""
        self.add_item(MISBTag.SENSOR_LONGITUDE,
                      self.encoder.encode_longitude(lon))

    def add_sensor_altitude(self, alt: float):
        """Add sensor true altitude (Tag 15)."""
        self.add_item(MISBTag.SENSOR_TRUE_ALTITUDE,
                      self.encoder.encode_altitude(alt))

    def add_sensor_horizontal_fov(self, fov: float):
        """Add sensor horizontal FOV (Tag 16)."""
        self.add_item(MISBTag.SENSOR_HORIZONTAL_FOV,
                      self.encoder.encode_fov(fov))

    def add_sensor_vertical_fov(self, fov: float):
        """Add sensor vertical FOV (Tag 17)."""
        self.add_item(MISBTag.SENSOR_VERTICAL_FOV,
                      self.encoder.encode_fov(fov))

    def add_sensor_rel_azimuth(self, azimuth: float):
        """Add sensor relative azimuth (Tag 18)."""
        self.add_item(MISBTag.SENSOR_REL_AZIMUTH,
                      self.encoder.encode_sensor_rel_angle(azimuth))

    def add_sensor_rel_elevation(self, elevation: float):
        """Add sensor relative elevation (Tag 19)."""
        self.add_item(MISBTag.SENSOR_REL_ELEVATION,
                      self.encoder.encode_sensor_rel_angle(elevation + 180))  # Offset for negative values

    def add_sensor_rel_roll(self, roll: float):
        """Add sensor relative roll (Tag 20)."""
        self.add_item(MISBTag.SENSOR_REL_ROLL,
                      self.encoder.encode_sensor_rel_angle(roll + 180))  # Offset for negative values

    def add_frame_center_latitude(self, lat: float):
        """Add frame center latitude (Tag 23)."""
        self.add_item(MISBTag.FRAME_CENTER_LATITUDE,
                      self.encoder.encode_latitude(lat))

    def add_frame_center_longitude(self, lon: float):
        """Add frame center longitude (Tag 24)."""
        self.add_item(MISBTag.FRAME_CENTER_LONGITUDE,
                      self.encoder.encode_longitude(lon))

    def add_uas_lds_version(self, version: int = 17):
        """Add UAS LDS version number (Tag 65)."""
        self.add_item(MISBTag.UAS_LDS_VERSION,
                      self.encoder.encode_uint8(version))

    def build(self) -> bytes:
        """
        Build the complete KLV packet with checksum.

        Returns:
            Complete MISB ST 0601 packet ready for embedding.
        """
        # Build the value portion (all tag-length-value triplets)
        value_data = bytearray()

        for tag, value in self.items:
            # Encode tag (BER-OID)
            value_data.extend(self.encoder.encode_ber_oid_length(tag))
            # Encode length (BER)
            value_data.extend(self.encoder.encode_ber_length(len(value)))
            # Add value
            value_data.extend(value)

        # Calculate checksum over UL + length + value
        # First, build packet without checksum to calculate it
        temp_packet = bytearray(MISB_UAS_LOCAL_SET_UL)
        # Add 2 bytes for checksum tag + 2 bytes for checksum value
        total_length = len(value_data) + 4
        temp_packet.extend(self.encoder.encode_ber_length(total_length))
        temp_packet.extend(value_data)

        # Add checksum placeholder for calculation
        temp_packet.append(MISBTag.CHECKSUM)  # Tag 1
        temp_packet.append(2)  # Length = 2 bytes
        temp_packet.extend([0, 0])  # Placeholder

        # Calculate checksum
        checksum = self.encoder.calculate_checksum(bytes(temp_packet))

        # Build final packet
        packet = bytearray(MISB_UAS_LOCAL_SET_UL)
        packet.extend(self.encoder.encode_ber_length(total_length))
        packet.extend(value_data)

        # Add checksum (Tag 1)
        packet.append(MISBTag.CHECKSUM)
        packet.append(2)
        packet.extend(struct.pack('>H', checksum))

        return bytes(packet)


class MISBVideoConverter:
    """
    Converter for creating MP4 videos with embedded MISB ST 0601 KLV metadata.
    """

    def __init__(
            self,
            sequence_id: str,
            images_folder: str,
            poses_file: str,
            output_path: str,
            origin_file: Optional[str] = None,
            fps: float = 30.0,
            video_codec: str = "libx264",
            crf: int = 23,
            mission_id: Optional[str] = None,
            platform_designation: Optional[str] = None,
            verbose: bool = True
    ):
        """
        Initialize the MISB video converter.

        Args:
            sequence_id: Sequence identifier
            images_folder: Folder containing frame images
            poses_file: Path to poses JSON file
            output_path: Output MP4 file path
            origin_file: Optional DEM metadata file with origin coordinates
            fps: Output video frame rate
            video_codec: FFmpeg video codec
            crf: Constant Rate Factor for encoding quality (lower = better)
            mission_id: Optional mission ID for metadata
            platform_designation: Optional platform designation
            verbose: Whether to print progress
        """
        self.sequence_id = sequence_id
        self.images_folder = images_folder
        self.poses_file = poses_file
        self.output_path = output_path
        self.origin_file = origin_file
        self.fps = fps
        self.video_codec = video_codec
        self.crf = crf
        self.mission_id = mission_id or f"Mission_{sequence_id}"
        self.platform_designation = platform_designation
        self.verbose = verbose

        # Data containers
        self.poses: List[FramePose] = []
        self.origin_lat: float = 0.0
        self.origin_lon: float = 0.0
        self.origin_alt: float = 0.0
        self.drone_name: str = "Unknown"
        self.camera_name: str = "Unknown"

        # Frame dimensions (set when loading first frame)
        self.frame_width: int = 0
        self.frame_height: int = 0

        # Load data
        self._load_poses()

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _load_poses(self):
        """Load pose data from JSON file."""
        self._log(f"Loading poses from: {self.poses_file}")

        if not os.path.exists(self.poses_file):
            raise FileNotFoundError(f"Poses file not found: {self.poses_file}")

        with open(self.poses_file, 'r') as f:
            data = json.load(f)

        # Extract origin
        origin = data.get("origin", {})
        self.origin_lat = origin.get("latitude", 0.0)
        self.origin_lon = origin.get("longitude", 0.0)
        self.origin_alt = origin.get("altitude", 0.0)

        # Extract platform info
        self.drone_name = data.get("drone", "Unknown UAV")
        self.camera_name = data.get("camera", "Unknown Camera")

        if self.platform_designation is None:
            self.platform_designation = self.drone_name

        # Parse image poses
        images = data.get("images", [])
        self._log(f"  Found {len(images)} frames")

        for img_data in images:
            # Parse timestamp
            timestamp_str = img_data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Fallback to current time if parsing fails
                timestamp = datetime.now(timezone.utc)

            # Get coordinates (prefer lat/lng if available, otherwise use local + origin)
            lat = img_data.get("lat", self.origin_lat)
            lon = img_data.get("lng", self.origin_lon)

            # Get altitude from location[2] + origin altitude
            location = img_data.get("location", [0, 0, 0])
            alt = location[2] + self.origin_alt if len(location) > 2 else self.origin_alt

            # Parse rotation (assuming [pitch, roll, yaw] or similar)
            rotation = img_data.get("rotation", [0, 0, 0])
            # The rotation format from your data appears to be [pitch?, roll?, yaw/heading]
            # Based on the data, rotation[2] changes gradually which suggests heading
            pitch = rotation[0] if len(rotation) > 0 else 0
            roll = rotation[1] if len(rotation) > 1 else 0
            heading = rotation[2] if len(rotation) > 2 else 0

            # Normalize heading to 0-360
            heading = heading % 360.0

            # Normalize pitch (appears to be 360 in your data, which should be 0)
            if pitch >= 359:
                pitch = pitch - 360

            # Get FOV
            fov_data = img_data.get("fovy", [45.0])
            fov_v = fov_data[0] if isinstance(fov_data, list) else fov_data
            # Estimate horizontal FOV assuming 16:9 aspect ratio
            fov_h = fov_v * 16 / 9

            pose = FramePose(
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                heading=heading,
                pitch=pitch,
                roll=roll,
                fov_horizontal=fov_h,
                fov_vertical=fov_v,
                imagefile=img_data.get("imagefile", "")
            )
            self.poses.append(pose)

        self._log(f"  Origin: ({self.origin_lat:.6f}, {self.origin_lon:.6f}, {self.origin_alt:.1f}m)")
        self._log(f"  Platform: {self.drone_name}, Camera: {self.camera_name}")

        if self.poses:
            self._log(f"  Time range: {self.poses[0].timestamp} to {self.poses[-1].timestamp}")

    def _find_image_path(self, pose: FramePose, frame_idx: int) -> Optional[str]:
        """Find the image file for a frame."""
        image_filename = pose.imagefile

        possible_paths = [
            os.path.join(self.images_folder, self.sequence_id, "img1", image_filename),
            os.path.join(self.images_folder, "img1", image_filename),
            os.path.join(self.images_folder, self.sequence_id, image_filename),
            os.path.join(self.images_folder, image_filename),
            os.path.join(self.images_folder, self.sequence_id, "img1", f"{frame_idx:08d}.png"),
            os.path.join(self.images_folder, self.sequence_id, "img1", f"{frame_idx:08d}.jpg"),
            os.path.join(self.images_folder, f"{frame_idx:08d}.png"),
            os.path.join(self.images_folder, f"{frame_idx:08d}.jpg"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _build_klv_packet(self, pose: FramePose) -> bytes:
        """Build a KLV packet for a frame pose."""
        builder = MISBPacketBuilder()

        # Add version first
        builder.add_uas_lds_version(17)

        # Add timestamp
        builder.add_timestamp(pose.timestamp)

        # Add mission/platform info
        builder.add_mission_id(self.mission_id)
        builder.add_platform_designation(self.platform_designation)
        builder.add_platform_tail_number(self.sequence_id)
        builder.add_image_source_sensor(self.camera_name)

        # Add platform attitude
        builder.add_platform_heading(pose.heading)
        builder.add_platform_pitch(pose.pitch)
        builder.add_platform_roll(pose.roll)

        # Add sensor position
        builder.add_sensor_latitude(pose.latitude)
        builder.add_sensor_longitude(pose.longitude)
        builder.add_sensor_altitude(pose.altitude)

        # Add sensor FOV
        builder.add_sensor_horizontal_fov(pose.fov_horizontal)
        builder.add_sensor_vertical_fov(pose.fov_vertical)

        # Add sensor orientation (relative to platform)
        # For nadir-looking camera, azimuth=0, elevation=-90, roll=0
        builder.add_sensor_rel_azimuth(0.0)
        builder.add_sensor_rel_elevation(-90.0)  # Looking down
        builder.add_sensor_rel_roll(0.0)

        # Add frame center (same as sensor position for nadir view)
        builder.add_frame_center_latitude(pose.latitude)
        builder.add_frame_center_longitude(pose.longitude)

        return builder.build()

    def _write_klv_file(self, output_path: str, start_frame: int = 0,
                        end_frame: Optional[int] = None, step: int = 1) -> str:
        """
        Write KLV data to a binary file for muxing.

        Returns path to the KLV file.
        """
        if end_frame is None:
            end_frame = len(self.poses)

        klv_path = output_path.rsplit('.', 1)[0] + '.klv'

        with open(klv_path, 'wb') as f:
            for i in range(start_frame, min(end_frame, len(self.poses)), step):
                pose = self.poses[i]
                packet = self._build_klv_packet(pose)
                f.write(packet)

        return klv_path

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def export_to_kml(self, output_path: str = None):
        """
        Export flight path to KML for Google Earth.
        """
        if not output_path:
            output_path = self.output_path.rsplit('.', 1)[0] + '.kml'

        header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Drone Flight Path</name>
    <Style id="yellowLineGreenPoly">
      <LineStyle>
        <color>7f00ffff</color>
        <width>4</width>
      </LineStyle>
      <PolyStyle>
        <color>7f00ff00</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#yellowLineGreenPoly</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
"""
        footer = """        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""

        self._log(f"Exporting KML to: {output_path}")
        with open(output_path, 'w') as f:
            f.write(header)
            for pose in self.poses:
                # KML format: lon,lat,alt
                f.write(f"          {pose.longitude},{pose.latitude},{pose.altitude}\n")
            f.write(footer)

        return output_path

    def export_to_csv(self, output_path: str = None):
        """
        Export metadata to CSV for QGIS Temporal Controller.
        """
        if not output_path:
            output_path = self.output_path.rsplit('.', 1)[0] + '.csv'

        self._log(f"Exporting CSV to: {output_path}")
        with open(output_path, 'w') as f:
            # Header
            f.write("timestamp,latitude,longitude,altitude,heading,pitch,roll,image_file\n")
            for pose in self.poses:
                # Format timestamp for QGIS (ISO 8601)
                ts_str = pose.timestamp.isoformat()
                f.write(f"{ts_str},{pose.latitude},{pose.longitude},{pose.altitude},"
                        f"{pose.heading},{pose.pitch},{pose.roll},{pose.imagefile}\n")

        return output_path

    def convert(
            self,
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            step: int = 1
    ) -> Optional[str]:
        """
        Convert frames to MPEG-TS video with embedded KLV metadata (STANAG 4609 style).
        """
        if not self._check_ffmpeg():
            self._log("Error: FFmpeg not found. Please install FFmpeg.")
            return None

        if end_frame is None:
            end_frame = len(self.poses)

        frame_indices = list(range(start_frame, min(end_frame, len(self.poses)), step))
        n_frames = len(frame_indices)

        # Force output to be .ts (MPEG Transport Stream)
        # This is the standard container for drone video with KLV (STANAG 4609)
        final_output_path = self.output_path
        if final_output_path.lower().endswith('.mp4'):
            final_output_path = final_output_path[:-4] + '.ts'

        self._log(f"\nConverting {n_frames} frames to MISB video")
        self._log(f"  Output: {final_output_path}")
        self._log(f"  FPS: {self.fps}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Prepare frames
            self._log("\nStep 1: Preparing frames...")
            frame_list_path = os.path.join(temp_dir, 'frames.txt')
            klv_packets = []
            copied_frames = []

            for out_idx, frame_idx in enumerate(frame_indices):
                pose = self.poses[frame_idx]
                src_path = self._find_image_path(pose, frame_idx)

                if src_path is None:
                    continue

                if self.frame_width == 0:
                    img = cv2.imread(src_path)
                    if img is not None:
                        self.frame_height, self.frame_width = img.shape[:2]

                frame_path = os.path.join(temp_dir, f"frame_{len(copied_frames):08d}.png")

                try:
                    shutil.copy2(src_path, frame_path)
                    copied_frames.append(frame_path)
                    klv_packets.append(self._build_klv_packet(pose))
                except Exception:
                    continue

            if not copied_frames:
                self._log("Error: No frames found.")
                return None

            # Write frame list
            with open(frame_list_path, 'w') as frame_list:
                for frame_path in copied_frames:
                    safe_path = frame_path.replace('\\', '/')
                    frame_list.write(f"file '{safe_path}'\n")
                    frame_list.write(f"duration {1.0 / self.fps}\n")

            # Step 2: Write raw KLV data
            self._log("\nStep 2: Writing KLV metadata...")
            klv_path = os.path.join(temp_dir, 'metadata.klv')
            with open(klv_path, 'wb') as f:
                for packet in klv_packets:
                    f.write(packet)

            # Step 3: Encode video to intermediate file
            self._log("\nStep 3: Encoding video...")
            temp_video = os.path.join(temp_dir, 'temp_video.mp4')

            ffmpeg_video_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', frame_list_path,
                '-c:v', self.video_codec, '-crf', str(self.crf),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-r', str(self.fps),
                temp_video
            ]

            subprocess.run(ffmpeg_video_cmd, capture_output=True)

            if not os.path.exists(temp_video):
                self._log("  Video encoding failed.")
                return None

            # Step 4: Mux to MPEG-TS
            self._log("\nStep 4: Muxing to MPEG-TS (STANAG 4609)...")

            output_dir = os.path.dirname(os.path.abspath(final_output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Mux command: Combine Video and KLV into .ts
            mux_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,  # Input 0: Video
                '-f', 'data',  # Force format for input 1
                '-i', klv_path,  # Input 1: Raw KLV
                '-map', '0:v',  # Map video from input 0
                '-map', '1:0',  # Map data from input 1
                '-c', 'copy',  # Copy both streams (don't re-encode)
                '-f', 'mpegts',  # Force MPEG-TS container
                final_output_path
            ]

            self._log(f"  Running Mux: {' '.join(mux_cmd)}")
            result = subprocess.run(mux_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self._log(f"  FFmpeg mux error: {result.stderr}")
                return None

            # Save KLV sidecar as well
            klv_sidecar = final_output_path.rsplit('.', 1)[0] + '.klv'
            shutil.copy2(klv_path, klv_sidecar)

        self._log(f"\n{'=' * 60}")
        self._log(f"Conversion complete!")
        self._log(f"  Video: {final_output_path}")
        self._log(f"  Sidecar KLV: {klv_sidecar}")

        return final_output_path

    def export_klv_only(
            self,
            output_path: Optional[str] = None,
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            step: int = 1
    ) -> str:
        """
        Export only the KLV metadata file without video.

        Useful for creating sidecar metadata files.
        """
        if output_path is None:
            output_path = self.output_path.rsplit('.', 1)[0] + '.klv'

        if end_frame is None:
            end_frame = len(self.poses)

        frame_indices = list(range(start_frame, min(end_frame, len(self.poses)), step))

        self._log(f"Exporting KLV metadata for {len(frame_indices)} frames")

        with open(output_path, 'wb') as f:
            for frame_idx in frame_indices:
                pose = self.poses[frame_idx]
                packet = self._build_klv_packet(pose)
                f.write(packet)

        self._log(f"KLV file saved: {output_path}")
        self._log(f"File size: {os.path.getsize(output_path)} bytes")

        return output_path

    def create_qgis_compatible_video(
            self,
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            step: int = 1
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Create video and associated files optimized for QGIS.

        QGIS typically reads KLV from a sidecar file or embedded stream.
        This method creates both for maximum compatibility.

        Returns:
            Tuple of (video_path, klv_path)
        """
        video_path = self.convert(start_frame, end_frame, step)
        klv_path = self.output_path.rsplit('.', 1)[0] + '.klv'

        if video_path and os.path.exists(klv_path):
            # Create a simple metadata JSON for additional QGIS compatibility
            meta_path = self.output_path.rsplit('.', 1)[0] + '_meta.json'

            meta = {
                "type": "MISB_ST_0601",
                "version": "0601.17",
                "video_file": os.path.basename(video_path),
                "klv_file": os.path.basename(klv_path),
                "fps": self.fps,
                "frame_count": len(self.poses),
                "crs": "EPSG:4326",
                "origin": {
                    "latitude": self.origin_lat,
                    "longitude": self.origin_lon,
                    "altitude": self.origin_alt
                },
                "platform": {
                    "designation": self.platform_designation,
                    "drone": self.drone_name,
                    "camera": self.camera_name
                },
                "time_range": {
                    "start": self.poses[0].timestamp.isoformat() if self.poses else None,
                    "end": self.poses[-1].timestamp.isoformat() if self.poses else None
                }
            }

            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            self._log(f"  Metadata JSON: {meta_path}")

        return video_path, klv_path


def main():
    parser = argparse.ArgumentParser(
        description="Create MP4 video with MISB ST 0601 KLV metadata from drone frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python misb_video_converter.py \\
        --sequence-id 14_1 \\
        --images-folder /path/to/sequences/test \\
        --poses-file /path/to/14_matched_poses.json \\
        --output /path/to/output.mp4

QGIS Usage:
    1. Open QGIS and add the generated video via Layer > Add Layer > Add... 
    2. Use the Video Player plugin or FMV plugin for playback
    3. The KLV metadata provides geolocation for each frame
        """
    )

    parser.add_argument('--sequence-id', '-s', type=str, default="14_1",
                        help='Sequence identifier (e.g., "14_1")')
    parser.add_argument('--images-folder', '-i', type=str, default=r"Z:\sequences\test",
                        help='Folder containing frame images')
    parser.add_argument('--poses-file', '-p', type=str, default=r"Z:\correction_data\14_matched_poses.json",
                        help='Path to poses JSON file')
    parser.add_argument('--output', '-o', type=str, default=r"Z:\misb\misb.mp4",
                        help='Output MP4 file path')

    parser.add_argument('--fps', type=float, default=30.0,
                        help='Output video frame rate (default: 30)')
    parser.add_argument('--codec', type=str, default='libx264',
                        help='Video codec (default: libx264)')
    parser.add_argument('--crf', type=int, default=23,
                        help='Constant Rate Factor for quality (default: 23, lower=better)')

    parser.add_argument('--mission-id', type=str, default=None,
                        help='Mission ID for metadata')
    parser.add_argument('--platform', type=str, default=None,
                        help='Platform designation for metadata')

    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame to process (default: 0)')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Last frame to process (default: all)')
    parser.add_argument('--step', type=int, default=1,
                        help='Process every Nth frame (default: 1)')

    parser.add_argument('--klv-only', action='store_true',
                        help='Export only KLV metadata without video')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.images_folder):
        print(f"Error: Images folder does not exist: {args.images_folder}")
        sys.exit(1)

    if not os.path.isfile(args.poses_file):
        print(f"Error: Poses file does not exist: {args.poses_file}")
        sys.exit(1)

    # Create converter
    converter = MISBVideoConverter(
        sequence_id=args.sequence_id,
        images_folder=args.images_folder,
        poses_file=args.poses_file,
        output_path=args.output,
        fps=args.fps,
        video_codec=args.codec,
        crf=args.crf,
        mission_id=args.mission_id,
        platform_designation=args.platform,
        verbose=not args.quiet
    )

    # Process
    if args.klv_only:
        klv_path = converter.export_klv_only(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step
        )
        print(f"\nGenerated KLV file: {klv_path}")
    else:
        video_path, klv_path = converter.create_qgis_compatible_video(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step
        )

        if video_path:
            # 1. Export KML for Google Earth
            converter.export_to_kml()

            # 2. Export CSV for QGIS
            csv_path = converter.export_to_csv()

            print(f"\nGenerated Files:")
            print(f"  Video: {video_path}")
            print(f"  KML (Google Earth): {video_path.replace('.ts', '.kml').replace('.mp4', '.kml')}")
            print(f"  CSV (QGIS Data):    {csv_path}")

            print(f"\nGenerated video: {video_path}")
            print(f"Generated KLV:   {klv_path}")
        else:
            print("\nConversion failed")
            sys.exit(1)


if __name__ == "__main__":
    main()