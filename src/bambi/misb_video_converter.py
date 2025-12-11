#!/usr/bin/env python3
"""
MISB ST 0601 Video Converter with DEM Support (v7b)

This version fixes KLV timing by:
1. Creating video first
2. Writing KLV packets with explicit timing markers
3. Using FFmpeg's -itsoffset and proper stream mapping

The key insight: FFmpeg needs to know the KLV data rate matches video frame rate.
We achieve this by writing KLV as a "video-like" stream.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import shutil
import math


def install_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        return __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--break-system-packages"])
        return __import__(import_name)


np = install_package("numpy")
cv2 = install_package("opencv-python", "cv2")
trimesh = install_package("trimesh")

try:
    from pyproj import Transformer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyproj", "--break-system-packages"])
    from pyproj import Transformer

try:
    from pyrr import Quaternion
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyrr", "--break-system-packages"])
    from pyrr import Quaternion

try:
    from scipy.signal import savgol_filter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--break-system-packages"])
    from scipy.signal import savgol_filter

MISB_UAS_LOCAL_SET_UL = bytes([
    0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
    0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00
])


class MISBTag:
    CHECKSUM = 1
    PRECISION_TIME_STAMP = 2
    MISSION_ID = 3
    PLATFORM_TAIL_NUMBER = 4
    PLATFORM_HEADING = 5
    PLATFORM_PITCH = 6
    PLATFORM_ROLL = 7
    PLATFORM_DESIGNATION = 10
    IMAGE_SOURCE_SENSOR = 11
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
    UAS_LDS_VERSION = 65
    CORNER_LAT_1_FULL = 82
    CORNER_LON_1_FULL = 83
    CORNER_LAT_2_FULL = 84
    CORNER_LON_2_FULL = 85
    CORNER_LAT_3_FULL = 86
    CORNER_LON_3_FULL = 87
    CORNER_LAT_4_FULL = 88
    CORNER_LON_4_FULL = 89


@dataclass
class FramePose:
    timestamp: datetime
    frame_index: int
    local_x: float
    local_y: float
    local_z: float
    latitude: float
    longitude: float
    altitude_msl: float
    heading: float
    pitch: float
    roll: float
    fov_horizontal: float
    fov_vertical: float
    rotation_matrix: np.ndarray
    imagefile: str
    raw_rotation: np.ndarray


@dataclass
class GroundIntersection:
    hit: bool
    local_x: float = 0.0
    local_y: float = 0.0
    local_z: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    slant_range: float = 0.0


@dataclass
class CornerPoints:
    lat1: float
    lon1: float
    lat2: float
    lon2: float
    lat3: float
    lon3: float
    lat4: float
    lon4: float


class KLVEncoder:
    @staticmethod
    def encode_ber_length(length: int) -> bytes:
        if length < 128:
            return bytes([length])
        length_bytes = []
        temp = length
        while temp > 0:
            length_bytes.insert(0, temp & 0xFF)
            temp >>= 8
        return bytes([0x80 | len(length_bytes)] + length_bytes)

    @staticmethod
    def encode_ber_oid_length(length: int) -> bytes:
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
        return struct.pack('>B', max(0, min(255, int(value))))

    @staticmethod
    def encode_string(value: str) -> bytes:
        return value.encode('utf-8')

    @staticmethod
    def encode_latitude(lat: float) -> bytes:
        lat = max(-90.0, min(90.0, lat))
        scaled = int(lat * (2 ** 31 - 1) / 90.0)
        return struct.pack('>i', scaled)

    @staticmethod
    def encode_longitude(lon: float) -> bytes:
        while lon > 180: lon -= 360
        while lon < -180: lon += 360
        scaled = int(lon * (2 ** 31 - 1) / 180.0)
        return struct.pack('>i', scaled)

    @staticmethod
    def encode_altitude(alt: float) -> bytes:
        alt = max(-900.0, min(19000.0, alt))
        scaled = int((alt + 900.0) * (2 ** 16 - 1) / 19900.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_heading(heading: float) -> bytes:
        heading = heading % 360.0
        scaled = int(heading * (2 ** 16 - 1) / 360.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_platform_pitch(pitch: float) -> bytes:
        pitch = max(-20.0, min(20.0, pitch))
        scaled = int((pitch + 20.0) * (2 ** 16 - 1) / 40.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_platform_roll(roll: float) -> bytes:
        roll = max(-50.0, min(50.0, roll))
        scaled = int((roll + 50.0) * (2 ** 16 - 1) / 100.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_sensor_rel_azimuth(azimuth: float) -> bytes:
        azimuth = azimuth % 360.0
        scaled = int(azimuth * (2 ** 32 - 1) / 360.0)
        return struct.pack('>I', scaled)

    @staticmethod
    def encode_sensor_rel_elevation(elevation: float) -> bytes:
        elevation = max(-180.0, min(180.0, elevation))
        scaled = int((elevation + 180.0) * (2 ** 32 - 1) / 360.0)
        return struct.pack('>I', scaled)

    @staticmethod
    def encode_sensor_rel_roll(roll: float) -> bytes:
        roll = roll % 360.0
        scaled = int(roll * (2 ** 32 - 1) / 360.0)
        return struct.pack('>I', scaled)

    @staticmethod
    def encode_fov(fov: float) -> bytes:
        fov = max(0.0, min(180.0, fov))
        scaled = int(fov * (2 ** 16 - 1) / 180.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_slant_range(range_m: float) -> bytes:
        range_m = max(0.0, min(5000000.0, range_m))
        scaled = int(range_m * (2 ** 32 - 1) / 5000000.0)
        return struct.pack('>I', scaled)

    @staticmethod
    def encode_target_width(width_m: float) -> bytes:
        width_m = max(0.0, min(10000.0, width_m))
        scaled = int(width_m * (2 ** 16 - 1) / 10000.0)
        return struct.pack('>H', scaled)

    @staticmethod
    def encode_timestamp(dt: datetime) -> bytes:
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        timestamp_us = int(dt.timestamp() * 1_000_000)
        return struct.pack('>Q', timestamp_us)

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        checksum = 0
        for i, byte in enumerate(data):
            checksum += byte << (8 * ((i + 1) % 2))
        return checksum & 0xFFFF


class MISBPacketBuilder:
    def __init__(self):
        self.items: List[Tuple[int, bytes]] = []
        self.encoder = KLVEncoder()

    def add_item(self, tag: int, value: bytes):
        self.items.append((tag, value))

    def add_timestamp(self, dt: datetime):
        self.add_item(MISBTag.PRECISION_TIME_STAMP, self.encoder.encode_timestamp(dt))

    def add_mission_id(self, mission_id: str):
        self.add_item(MISBTag.MISSION_ID, self.encoder.encode_string(mission_id))

    def add_platform_tail_number(self, tail_number: str):
        self.add_item(MISBTag.PLATFORM_TAIL_NUMBER, self.encoder.encode_string(tail_number))

    def add_platform_heading(self, heading: float):
        self.add_item(MISBTag.PLATFORM_HEADING, self.encoder.encode_heading(heading))

    def add_platform_pitch(self, pitch: float):
        self.add_item(MISBTag.PLATFORM_PITCH, self.encoder.encode_platform_pitch(pitch))

    def add_platform_roll(self, roll: float):
        self.add_item(MISBTag.PLATFORM_ROLL, self.encoder.encode_platform_roll(roll))

    def add_platform_designation(self, designation: str):
        self.add_item(MISBTag.PLATFORM_DESIGNATION, self.encoder.encode_string(designation))

    def add_image_source_sensor(self, sensor: str):
        self.add_item(MISBTag.IMAGE_SOURCE_SENSOR, self.encoder.encode_string(sensor))

    def add_sensor_latitude(self, lat: float):
        self.add_item(MISBTag.SENSOR_LATITUDE, self.encoder.encode_latitude(lat))

    def add_sensor_longitude(self, lon: float):
        self.add_item(MISBTag.SENSOR_LONGITUDE, self.encoder.encode_longitude(lon))

    def add_sensor_altitude(self, alt: float):
        self.add_item(MISBTag.SENSOR_TRUE_ALTITUDE, self.encoder.encode_altitude(alt))

    def add_sensor_horizontal_fov(self, fov: float):
        self.add_item(MISBTag.SENSOR_HORIZONTAL_FOV, self.encoder.encode_fov(fov))

    def add_sensor_vertical_fov(self, fov: float):
        self.add_item(MISBTag.SENSOR_VERTICAL_FOV, self.encoder.encode_fov(fov))

    def add_sensor_rel_azimuth(self, azimuth: float):
        self.add_item(MISBTag.SENSOR_REL_AZIMUTH, self.encoder.encode_sensor_rel_azimuth(azimuth))

    def add_sensor_rel_elevation(self, elevation: float):
        self.add_item(MISBTag.SENSOR_REL_ELEVATION, self.encoder.encode_sensor_rel_elevation(elevation))

    def add_sensor_rel_roll(self, roll: float):
        self.add_item(MISBTag.SENSOR_REL_ROLL, self.encoder.encode_sensor_rel_roll(roll))

    def add_slant_range(self, range_m: float):
        self.add_item(MISBTag.SLANT_RANGE, self.encoder.encode_slant_range(range_m))

    def add_target_width(self, width_m: float):
        self.add_item(MISBTag.TARGET_WIDTH, self.encoder.encode_target_width(width_m))

    def add_frame_center_latitude(self, lat: float):
        self.add_item(MISBTag.FRAME_CENTER_LATITUDE, self.encoder.encode_latitude(lat))

    def add_frame_center_longitude(self, lon: float):
        self.add_item(MISBTag.FRAME_CENTER_LONGITUDE, self.encoder.encode_longitude(lon))

    def add_frame_center_elevation(self, elev: float):
        if abs(elev) < 0.1:
            elev = 0.1
        self.add_item(MISBTag.FRAME_CENTER_ELEVATION, self.encoder.encode_altitude(elev))

    def add_corner_coordinates(self, corners: CornerPoints):
        self.add_item(MISBTag.CORNER_LAT_1_FULL, self.encoder.encode_latitude(corners.lat1))
        self.add_item(MISBTag.CORNER_LON_1_FULL, self.encoder.encode_longitude(corners.lon1))
        self.add_item(MISBTag.CORNER_LAT_2_FULL, self.encoder.encode_latitude(corners.lat2))
        self.add_item(MISBTag.CORNER_LON_2_FULL, self.encoder.encode_longitude(corners.lon2))
        self.add_item(MISBTag.CORNER_LAT_3_FULL, self.encoder.encode_latitude(corners.lat3))
        self.add_item(MISBTag.CORNER_LON_3_FULL, self.encoder.encode_longitude(corners.lon3))
        self.add_item(MISBTag.CORNER_LAT_4_FULL, self.encoder.encode_latitude(corners.lat4))
        self.add_item(MISBTag.CORNER_LON_4_FULL, self.encoder.encode_longitude(corners.lon4))

    def add_uas_lds_version(self, version: int = 17):
        self.add_item(MISBTag.UAS_LDS_VERSION, self.encoder.encode_uint8(version))

    def build(self) -> bytes:
        value_data = bytearray()
        for tag, value in self.items:
            value_data.extend(self.encoder.encode_ber_oid_length(tag))
            value_data.extend(self.encoder.encode_ber_length(len(value)))
            value_data.extend(value)

        temp_packet = bytearray(MISB_UAS_LOCAL_SET_UL)
        total_length = len(value_data) + 4
        temp_packet.extend(self.encoder.encode_ber_length(total_length))
        temp_packet.extend(value_data)
        temp_packet.append(MISBTag.CHECKSUM)
        temp_packet.append(2)
        temp_packet.extend([0, 0])

        checksum = self.encoder.calculate_checksum(bytes(temp_packet))

        packet = bytearray(MISB_UAS_LOCAL_SET_UL)
        packet.extend(self.encoder.encode_ber_length(total_length))
        packet.extend(value_data)
        packet.append(MISBTag.CHECKSUM)
        packet.append(2)
        packet.extend(struct.pack('>H', checksum))

        return bytes(packet)


class DEMRayCaster:
    def __init__(self, mesh_path: str, dem_meta_path: str, source_crs: str = "EPSG:32633"):
        self.mesh = None
        self.ray_caster = None
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.z_offset = 0.0

        self.transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

        self._load_mesh(mesh_path)
        self._load_dem_metadata(dem_meta_path)

    def _load_mesh(self, mesh_path: str):
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"DEM mesh not found: {mesh_path}")

        self.mesh = trimesh.load(mesh_path, force='mesh')

        try:
            self.ray_caster = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        except:
            self.ray_caster = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

        print(f"Loaded DEM: {len(self.mesh.vertices)} vertices")

    def _load_dem_metadata(self, dem_meta_path: str):
        if not os.path.exists(dem_meta_path):
            return
        with open(dem_meta_path, 'r') as f:
            meta = json.load(f)
        origin = meta.get("origin", [0, 0, 0])
        self.x_offset = origin[0]
        self.y_offset = origin[1]
        self.z_offset = origin[2]

    def local_to_wgs84(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        utm_x = x + self.x_offset
        utm_y = y + self.y_offset
        utm_z = z + self.z_offset
        lon, lat = self.transformer.transform(utm_x, utm_y)
        return (lat, lon, utm_z)

    def get_nadir_intersection(self, position: np.ndarray) -> GroundIntersection:
        result = GroundIntersection(hit=False)
        if self.ray_caster is None:
            return result

        direction = np.array([0.0, 0.0, -1.0])
        try:
            locations, _, _ = self.ray_caster.intersects_location(
                np.array([position]), np.array([direction]), multiple_hits=False
            )
        except:
            return result

        if len(locations) == 0:
            return result

        hit = locations[0]
        result.hit = True
        result.local_x = hit[0]
        result.local_y = hit[1]
        result.local_z = hit[2]
        result.slant_range = np.linalg.norm(hit - position)

        lat, lon, alt = self.local_to_wgs84(hit[0], hit[1], hit[2])
        result.latitude = lat
        result.longitude = lon
        result.elevation = alt

        return result


def destination_point(lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    R = 6371000
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    d_R = distance_m / R

    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(d_R) +
        math.cos(lat_rad) * math.sin(d_R) * math.cos(bearing_rad)
    )
    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(d_R) * math.cos(lat_rad),
        math.cos(d_R) - math.sin(lat_rad) * math.sin(lat2)
    )
    return (math.degrees(lat2), math.degrees(lon2))


def calculate_corner_points(center_lat, center_lon, heading_deg, target_width, target_height):
    half_w = target_width / 2.0
    half_h = target_height / 2.0
    diag = math.sqrt(half_w ** 2 + half_h ** 2)
    corner_angle = math.degrees(math.atan2(half_w, half_h))

    bearing_ul = (heading_deg - corner_angle) % 360
    lat1, lon1 = destination_point(center_lat, center_lon, diag, bearing_ul)

    bearing_ur = (heading_deg + corner_angle) % 360
    lat2, lon2 = destination_point(center_lat, center_lon, diag, bearing_ur)

    bearing_lr = (heading_deg + 180 - corner_angle) % 360
    lat3, lon3 = destination_point(center_lat, center_lon, diag, bearing_lr)

    bearing_ll = (heading_deg + 180 + corner_angle) % 360
    lat4, lon4 = destination_point(center_lat, center_lon, diag, bearing_ll)

    return CornerPoints(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2,
                        lat3=lat3, lon3=lon3, lat4=lat4, lon4=lon4)


class MISBVideoConverterDEM:
    def __init__(
            self,
            sequence_id: str,
            images_folder: str,
            data_folder: str,
            output_path: str,
            fps: float = 30.0,
            video_codec: str = "libx264",
            crf: int = 23,
            mission_id: Optional[str] = None,
            platform_designation: Optional[str] = None,
            source_crs: str = "EPSG:32633",
            apply_smoothing: bool = True,
            heading_offset: float = 0.0,
            swap_axes: bool = False,
            verbose: bool = True
    ):
        self.sequence_id = sequence_id
        self.images_folder = images_folder
        self.data_folder = data_folder
        self.output_path = output_path
        self.fps = fps
        self.video_codec = video_codec
        self.crf = crf
        self.mission_id = mission_id or f"Mission_{sequence_id}"
        self.platform_designation = platform_designation
        self.source_crs = source_crs
        self.apply_smoothing = apply_smoothing
        self.heading_offset = heading_offset
        self.swap_axes = swap_axes
        self.verbose = verbose

        self.poses: List[FramePose] = []
        self.translation: Dict = {'x': 0, 'y': 0, 'z': 0}
        self.rotation_correction: Dict = {'x': 0, 'y': 0, 'z': 0}

        self.drone_name = "Unknown"
        self.camera_name = "Unknown"
        self.frame_width = 0
        self.frame_height = 0
        self.aspect_ratio = 16.0 / 9.0

        self.dem: Optional[DEMRayCaster] = None
        self.base_timestamp = datetime.now(timezone.utc)

        self._load_data()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_data(self):
        self._log(f"Loading data for sequence: {self.sequence_id}")

        parent_id = self.sequence_id.split('_')[0]

        dem_path = os.path.join(self.data_folder, f"{parent_id}_dem.glb")
        dem_meta_path = os.path.join(self.data_folder, f"{parent_id}_dem_mesh_r2.json")
        poses_path = os.path.join(self.data_folder, f"{parent_id}_matched_poses.json")
        correction_path = os.path.join(self.data_folder, f"{parent_id}_correction.json")

        if not os.path.exists(poses_path):
            poses_path = os.path.join(self.data_folder, f"{self.sequence_id}_poses.json")

        self.dem = DEMRayCaster(dem_path, dem_meta_path, self.source_crs)
        self._load_correction(correction_path)
        self._load_poses(poses_path)

        self._log(f"Loaded {len(self.poses)} frames")
        if self.poses:
            self.base_timestamp = self.poses[0].timestamp

    def _load_correction(self, correction_path: str):
        if not os.path.exists(correction_path):
            self._log("No correction file found")
            return
        with open(correction_path, 'r') as f:
            correction = json.load(f)
        self.translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
        self.rotation_correction = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})

    def _calculate_heading_from_trajectory(self, idx: int, images: List[dict]) -> float:
        if idx >= len(images) - 1:
            idx = len(images) - 2
        if idx < 0:
            idx = 0

        loc1 = np.array(images[idx].get('location', [0, 0, 0]))
        loc2 = np.array(images[min(idx + 1, len(images) - 1)].get('location', [0, 0, 0]))

        dx = loc2[0] - loc1[0]
        dy = loc2[1] - loc1[1]

        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return 0.0

        if self.swap_axes:
            east_component = dy
            north_component = dx
        else:
            east_component = dx
            north_component = dy

        heading_rad = math.atan2(east_component, north_component)
        heading_deg = math.degrees(heading_rad)

        if heading_deg < 0:
            heading_deg += 360

        return heading_deg

    def _load_poses(self, poses_path: str):
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"Poses not found: {poses_path}")

        with open(poses_path, 'r') as f:
            data = json.load(f)

        self.drone_name = data.get("drone", "UAV")
        self.camera_name = data.get("camera", "Camera")
        if self.platform_designation is None:
            self.platform_designation = self.drone_name

        images = data.get("images", [])

        if self.apply_smoothing and len(images) >= 5:
            positions = np.array([img["location"] for img in images], dtype=float)
            wl = min(11, len(images) - 1 if (len(images) - 1) % 2 == 1 else len(images) - 2)
            if wl >= 3:
                smoothed = savgol_filter(positions, window_length=wl, polyorder=2, axis=0)
                for img, loc in zip(images, smoothed):
                    img["location"] = loc.tolist()

        for idx, img_data in enumerate(images):
            ts_str = img_data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(ts_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except:
                timestamp = datetime.now(timezone.utc) + timedelta(seconds=idx / self.fps)

            location = np.array(img_data.get('location', [0, 0, 0]), dtype=float)
            location += np.array([self.translation['x'], self.translation['y'], self.translation['z']])

            raw_rot = np.array(img_data.get('rotation', [0, 0, 0]), dtype=float)
            cor_rot = np.array(
                [self.rotation_correction['x'], self.rotation_correction['y'], self.rotation_correction['z']])

            rot_rad = np.deg2rad((raw_rot % 360.0) - cor_rot) * -1
            quat = Quaternion.from_eulers(rot_rad)
            rot_mat = np.array(quat.matrix33)

            heading = self._calculate_heading_from_trajectory(idx, images)
            heading = (heading + self.heading_offset) % 360.0

            pitch = raw_rot[0] - cor_rot[0]
            if pitch >= 359:
                pitch -= 360
            roll = raw_rot[1] - cor_rot[1]

            fov_data = img_data.get('fovy', [45.0])
            fov_v = fov_data[0] if isinstance(fov_data, list) else fov_data
            fov_h = fov_v * self.aspect_ratio

            lat, lon, alt = self.dem.local_to_wgs84(location[0], location[1], location[2])

            self.poses.append(FramePose(
                timestamp=timestamp, frame_index=idx,
                local_x=location[0], local_y=location[1], local_z=location[2],
                latitude=lat, longitude=lon, altitude_msl=alt,
                heading=heading, pitch=pitch, roll=roll,
                fov_horizontal=fov_h, fov_vertical=fov_v,
                rotation_matrix=rot_mat,
                imagefile=img_data.get("imagefile", ""),
                raw_rotation=raw_rot
            ))

    def _find_image_path(self, pose: FramePose, idx: int) -> Optional[str]:
        paths = [
            os.path.join(self.images_folder, self.sequence_id, "img1", pose.imagefile),
            os.path.join(self.images_folder, "img1", pose.imagefile),
            os.path.join(self.images_folder, self.sequence_id, pose.imagefile),
            os.path.join(self.images_folder, pose.imagefile),
            os.path.join(self.images_folder, self.sequence_id, "img1", f"{idx:08d}.png"),
            os.path.join(self.images_folder, self.sequence_id, "img1", f"{idx:08d}.jpg"),
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def _build_klv_packet(self, pose: FramePose, video_idx: int) -> bytes:
        builder = MISBPacketBuilder()

        frame_ts = self.base_timestamp + timedelta(seconds=video_idx / self.fps)

        builder.add_uas_lds_version(17)
        builder.add_timestamp(frame_ts)

        builder.add_mission_id(self.mission_id)
        builder.add_platform_designation(self.platform_designation)
        builder.add_platform_tail_number(self.sequence_id)
        builder.add_image_source_sensor(self.camera_name)

        builder.add_platform_heading(pose.heading)
        builder.add_platform_pitch(pose.pitch)
        builder.add_platform_roll(pose.roll)

        builder.add_sensor_latitude(pose.latitude)
        builder.add_sensor_longitude(pose.longitude)
        builder.add_sensor_altitude(pose.altitude_msl)

        builder.add_sensor_horizontal_fov(pose.fov_horizontal)
        builder.add_sensor_vertical_fov(pose.fov_vertical)

        builder.add_sensor_rel_azimuth(0.0)
        builder.add_sensor_rel_elevation(-90.0)
        builder.add_sensor_rel_roll(0.0)

        sensor_pos = np.array([pose.local_x, pose.local_y, pose.local_z])
        nadir = self.dem.get_nadir_intersection(sensor_pos)

        if nadir.hit:
            fc_lat = nadir.latitude
            fc_lon = nadir.longitude
            fc_elev = nadir.elevation
            slant_range = nadir.slant_range
        else:
            fc_lat = pose.latitude
            fc_lon = pose.longitude
            fc_elev = max(1.0, self.dem.z_offset)
            slant_range = max(10.0, pose.altitude_msl - fc_elev)

        target_width = 2.0 * slant_range * math.tan(math.radians(pose.fov_horizontal / 2.0))
        target_height = 2.0 * slant_range * math.tan(math.radians(pose.fov_vertical / 2.0))

        builder.add_slant_range(slant_range)
        builder.add_target_width(target_width)

        builder.add_frame_center_latitude(fc_lat)
        builder.add_frame_center_longitude(fc_lon)
        builder.add_frame_center_elevation(fc_elev)

        corners = calculate_corner_points(fc_lat, fc_lon, pose.heading, target_width, target_height)
        builder.add_corner_coordinates(corners)

        return builder.build()

    def convert(self, start_frame: int = 0, end_frame: Optional[int] = None, step: int = 1) -> Optional[str]:
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except:
            self._log("Error: FFmpeg not found!")
            return None

        if end_frame is None:
            end_frame = len(self.poses)

        frame_indices = list(range(start_frame, min(end_frame, len(self.poses)), step))
        self._log(f"\nConverting {len(frame_indices)} frames at {self.fps} fps")

        final_path = self.output_path
        if not final_path.endswith('.ts'):
            final_path = final_path.rsplit('.', 1)[0] + '.ts'

        with tempfile.TemporaryDirectory() as tmpdir:
            # Collect valid frames
            valid_frames = []
            for vid_idx, frm_idx in enumerate(frame_indices):
                pose = self.poses[frm_idx]
                src = self._find_image_path(pose, frm_idx)
                if src is None:
                    continue
                if self.frame_width == 0:
                    img = cv2.imread(src)
                    if img is not None:
                        self.frame_height, self.frame_width = img.shape[:2]
                        self.aspect_ratio = self.frame_width / self.frame_height
                        for p in self.poses:
                            p.fov_horizontal = p.fov_vertical * self.aspect_ratio
                valid_frames.append((vid_idx, frm_idx, pose, src))

            self._log(f"Found {len(valid_frames)} valid frames")
            if not valid_frames:
                return None

            # Create frame list for video
            frame_list = os.path.join(tmpdir, 'frames.txt')
            with open(frame_list, 'w') as f:
                for vid_idx, frm_idx, pose, src in valid_frames:
                    f.write(f"file '{src}'\n")
                    f.write(f"duration {1.0 / self.fps}\n")

            # Create video
            self._log("\nStep 1: Creating video...")
            temp_video = os.path.join(tmpdir, 'video.ts')
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', frame_list,
                '-c:v', self.video_codec, '-crf', str(self.crf),
                '-pix_fmt', 'yuv420p', '-r', str(self.fps),
                '-f', 'mpegts',
                temp_video
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self._log(f"Video error: {result.stderr}")
                return None

            # Create individual KLV files with proper sizes to match frame timing
            self._log("\nStep 2: Creating synchronized KLV stream...")

            # Calculate packet size to match frame rate
            # Each KLV packet should be associated with exactly one frame
            klv_packets = []
            for vid_idx, frm_idx, pose, src in valid_frames:
                packet = self._build_klv_packet(pose, vid_idx)
                klv_packets.append(packet)

            # Pad packets to same size for consistent timing
            max_size = max(len(p) for p in klv_packets)
            # Pad to next multiple of 188 (TS packet size) for cleaner muxing
            target_size = ((max_size // 188) + 1) * 188

            padded_klv_path = os.path.join(tmpdir, 'klv_padded.bin')
            with open(padded_klv_path, 'wb') as f:
                for packet in klv_packets:
                    f.write(packet)
                    # Add padding
                    padding_needed = target_size - len(packet)
                    if padding_needed > 0:
                        f.write(b'\xFF' * padding_needed)

            self._log(f"  Packet size: {target_size} bytes each")
            self._log(f"  Total KLV: {len(klv_packets)} packets")

            # Method: Use rawvideo-style approach for data stream
            # Treat KLV as fixed-size "frames" at video frame rate
            self._log("\nStep 3: Muxing with synchronized timestamps...")

            os.makedirs(os.path.dirname(os.path.abspath(final_path)) or '.', exist_ok=True)

            # First try: use -r to set data stream frame rate
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-f', 'rawvideo',
                '-video_size', f'{target_size}x1',
                '-pixel_format', 'gray8',
                '-r', str(self.fps),
                '-i', padded_klv_path,
                '-map', '0:v',
                '-map', '1:0',
                '-c:v', 'copy',
                '-c:d', 'copy',
                '-f', 'mpegts',
                final_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                # Fallback: Create TS by concatenating segments
                self._log("  Rawvideo method failed, using segment approach...")

                # Create individual segments
                segment_files = []
                for i, (vid_idx, frm_idx, pose, src) in enumerate(valid_frames):
                    seg_path = os.path.join(tmpdir, f'seg_{i:06d}.ts')
                    klv_path = os.path.join(tmpdir, f'klv_{i:06d}.bin')

                    with open(klv_path, 'wb') as f:
                        f.write(klv_packets[i])

                    cmd = [
                        'ffmpeg', '-y', '-v', 'error',
                        '-loop', '1', '-i', src,
                        '-f', 'data', '-i', klv_path,
                        '-t', str(1.0 / self.fps),
                        '-map', '0:v', '-map', '1:0',
                        '-c:v', self.video_codec, '-crf', str(self.crf),
                        '-pix_fmt', 'yuv420p',
                        '-f', 'mpegts',
                        seg_path
                    ]
                    subprocess.run(cmd, capture_output=True)
                    if os.path.exists(seg_path):
                        segment_files.append(seg_path)

                # Concatenate
                concat_list = os.path.join(tmpdir, 'concat.txt')
                with open(concat_list, 'w') as f:
                    for seg in segment_files:
                        f.write(f"file '{seg}'\n")

                cmd = [
                    'ffmpeg', '-y', '-v', 'error',
                    '-f', 'concat', '-safe', '0',
                    '-i', concat_list,
                    '-c', 'copy',
                    final_path
                ]
                subprocess.run(cmd, capture_output=True)

            # Create sidecar KLV
            klv_sidecar = final_path.rsplit('.', 1)[0] + '.klv'
            with open(klv_sidecar, 'wb') as f:
                for packet in klv_packets:
                    f.write(packet)

            # Verify timing
            self._log("\nStep 4: Verifying KLV timing...")
            cmd = ['ffprobe', '-v', 'error', '-show_entries',
                   'packet=pts_time', '-select_streams', 'd:0',
                   '-of', 'csv=p=0', final_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            pts_lines = [x.replace(",", "") for x in result.stdout.strip().split('\n') if x]

            if len(pts_lines) > 1:
                pts_times = [float(x) for x in pts_lines]
                self._log(f"  ✓ KLV packets: {len(pts_times)}")
                self._log(f"  ✓ PTS range: {min(pts_times):.2f}s - {max(pts_times):.2f}s")
                self._log(f"  ✓ Coverage: {max(pts_times) - min(pts_times):.2f}s")
            else:
                self._log(f"  ⚠ Limited KLV timing data")

        self._log(f"\n{'=' * 60}")
        self._log(f"Output: {final_path}")
        return final_path


def main():
    parser = argparse.ArgumentParser(description="MISB video converter (v7b)")
    parser.add_argument('--sequence-id', '-s', default="14_1")
    parser.add_argument('--images-folder', '-i', default=r"Z:\sequences\test")
    parser.add_argument('--data-folder', '-d', default=r"Z:\correction_data")
    parser.add_argument('--output', '-o', default=r"Z:\misb\misb.ts")
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--codec', default='libx264')
    parser.add_argument('--crf', type=int, default=23)
    parser.add_argument('--mission-id', default=None)
    parser.add_argument('--platform', default=None)
    parser.add_argument('--crs', default='EPSG:32633')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--no-smoothing', action='store_true')
    parser.add_argument('--heading-offset', type=float, default=0.0)
    parser.add_argument('--swap-axes', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    converter = MISBVideoConverterDEM(
        sequence_id=args.sequence_id,
        images_folder=args.images_folder,
        data_folder=args.data_folder,
        output_path=args.output,
        fps=args.fps,
        video_codec=args.codec,
        crf=args.crf,
        mission_id=args.mission_id,
        platform_designation=args.platform,
        source_crs=args.crs,
        apply_smoothing=not args.no_smoothing,
        heading_offset=args.heading_offset,
        swap_axes=args.swap_axes,
        verbose=not args.quiet
    )

    converter.convert(args.start_frame, args.end_frame, args.step)


if __name__ == "__main__":
    main()