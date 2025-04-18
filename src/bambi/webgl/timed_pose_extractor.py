import bisect
import datetime
import json
import os.path
from collections import defaultdict
from datetime import tzinfo
from itertools import islice
from numbers import Number
from typing import Any, List, Optional, Union, Final, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from attr import define
from dateutil import tz
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_interpolator import AirDataTimeInterpolator
from bambi.airdata.air_data_parser import AirDataParser
from bambi.domain.camera import Camera
from bambi.domain.drone import Drone
from bambi.domain.sensor import SensorResolution
from bambi.geo.gps_exif_writer import GpsExifWriter
from bambi.srt.srt_frame import SrtFrame
from bambi.srt.srt_parser import SrtParser
from bambi.util.image_utils import image_equality_check
from bambi.video.calibrated_video_frame_accessor import CalibratedVideoFrameAccessor
from bambi.video.video_frame_accessor import VideoFrameAccessor

TIMEZONE_VIENNA: Final[tzinfo] = tz.gettz('Europe/Vienna')
TIMEZONE_UTC: Final[tzinfo] = tz.tzutc()
AIR_DATA_MAX_FRAME_OFFSET: Final[int] = 2000  # milliseconds between frames
ORTHOMETRIC_TRANSFORMER = Transformer.from_crs(CRS("EPSG:4326").to_3d(), CRS("EPSG:4979").to_3d(),always_xy=True)  # WGS84 + EGM96 geoid height
# from geographiclib import Geoid

# Initialize Geoid model
# geoid = Geoid("egm96-5")

@define
class TimedFrameExtractorCallback:
    """Callback class that is used to store video frames and save them to disk together with their timestamps"""

    image_files: List[str]
    image_timestamps: List[datetime.datetime]
    srt_frames: List[SrtFrame]
    drone_name: Optional[Drone]
    camera_name: Optional[Camera]
    target_folder: str
    extension: str = "jpg"
    overwrite_existing: bool = False

    def __call__(self, idx: int, img: npt.NDArray[Any]) -> bool:

        if len(self.srt_frames) <= idx:
            return False  # skip

        frame = self.srt_frames[idx]
        frame_idx = frame.id
        assert frame.timestamp is not None
        timestamp = frame.timestamp

        filename = f"{len(self.image_files)}-{idx}-{frame_idx}.{self.extension}"
        path = os.path.join(self.target_folder, filename)
        if not os.path.exists(path) or self.overwrite_existing:
            cv2.imwrite(path, img)

        self.image_files.append(filename)
        self.image_timestamps.append(timestamp)

        return True


class TimedPoseExtractor:
    """
    Class that allows to extract video frames together with a JSON file describing the relative position between the first frame and the others
    """

    def __init__(
        self,
        calibrated_frame_accessor: VideoFrameAccessor,
        rel_transformer: Transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32633)),
        drone_name: Drone = Drone.M30T,
        camera_name: Camera = Camera.Thermal,
    ):
        """
        :param calibrated_frame_accessor: Object used to access the individual, normalized video frames
        :param rel_transformer: Used to calculate the relative position between from the reference point per frame
        :param drone_name: Drone used to create video
        :param camera_name: Camera used to create video
        """
        self.frame_accessor = calibrated_frame_accessor
        self.rel_transformer = rel_transformer
        self.drone_name = drone_name
        self.camera_name = camera_name


    @staticmethod
    def get_srt_frames(srt_files: Union[str, Sequence[str]], timezone: tzinfo = TIMEZONE_VIENNA) -> tuple[list[SrtFrame], int, list[int]]:
        """
        Loads all SRT frames from the given files and creates a video index lookup.
        :param srt_files: The SRT files to load.
        :param timezone: The timezone to assign to the frame timestamps.
        :return: A tuple containing all SRT frames, the number of video indices, and a list which maps a SRT frame index
        to the respective video index.
        """

        if isinstance(srt_files, str):
            srt_files = [srt_files]

        parser = SrtParser()

        srt_frames = []
        num_videos = 0
        frame_to_video = []

        for srt_file in srt_files:
            frames = parser.parse(srt_file)
            frame_count = len(frames)

            # apply local time zone
            for frame in frames:
                frame.timestamp = frame.timestamp.replace(tzinfo=timezone)

            srt_frames.extend(frames)
            frame_to_video.extend([num_videos] * frame_count)
            num_videos += 1

        return srt_frames, num_videos, frame_to_video


    @staticmethod
    def get_air_data_frames(air_data_file: str, video_timestamp: datetime.datetime,
                            timezone: tzinfo = TIMEZONE_VIENNA) -> Tuple[AirDataFrame, Sequence[AirDataFrame]]:
        """
        Loads all  AirData frames from an AirData CSV file and crops them to their video frames.
        :param air_data_file: The AirData CSV file to load.
        :param video_timestamp: The timestamp of the video creation.
        :param timezone: The timezone to assign to the frame timestamps.
        :return: A sequence of AirData frames representing the video frames.
        """
        ad_parser = AirDataParser()
        ad_frames = ad_parser.parse(air_data_file)

        start_frame = next(ad_frame for ad_frame in ad_frames if ad_frame is not None)
        start_date_utc = start_frame.datetime.replace(tzinfo=TIMEZONE_UTC)
        start_date = start_date_utc.astimezone(timezone)
        start_time = start_frame.time

        for frame in ad_frames:
            delta = frame.time - start_time
            frame.datetime = start_date + datetime.timedelta(milliseconds=delta)

        ms_offset = ad_parser.get_frames_video_offset(ad_frames, video_timestamp)

        # find the first and last isVideo frame (also consider ms_offset)
        is_videos = [bool(frame.isVideo) if (frame.time >= ms_offset) else False for frame in ad_frames]

        try:
            first_idx = is_videos.index(True)
        except ValueError:
            raise ValueError("AirData logs do not contain video frames")

        try:
            last_idx = is_videos.index(False, first_idx) - 1
        except ValueError:  # all frames after the first video frames are video frames
            last_idx = len(is_videos) - 1

        if last_idx == first_idx:
            raise ValueError(f"AirData logs do not form a valid video frame range [{first_idx}, {last_idx}]")

        return start_frame, ad_frames[first_idx: last_idx + 1]

    @staticmethod
    def _interpolate_nan(arr: npt.NDArray) -> npt.NDArray:
        valid_idx = np.where(~np.isnan(arr))[0]
        valid_values = arr[valid_idx]

        interpolator = interp1d(valid_idx, valid_values, kind="linear", fill_value="extrapolate")

        all_idx = np.arange(arr.size)
        arr = interpolator(all_idx)
        return arr

    @staticmethod
    def _get_srt_air_data_offset(srt_frames, ad_frames):
        # use minimization to find the best offset

        st = ad_frames[0].datetime  # start time
        # precompute (for speed)
        srt_seconds = np.array([(srt.timestamp - st).total_seconds() for srt in srt_frames])
        ad_seconds = np.array([(ad.datetime - st).total_seconds() for ad in ad_frames])
        ad_longs = np.array([frame.longitude for frame in ad_frames])
        ad_lats = np.array([frame.latitude for frame in ad_frames])

        has_nan_longs = False
        has_nan_lats = False
        frame_longs = []
        frame_lats = []
        for frame in srt_frames:
            longitude = frame.longitude
            if longitude is None:
                longitude = np.nan
                has_nan_longs = True
            frame_longs.append(longitude)

            latitude = frame.latitude
            if latitude is None:
                latitude = np.nan
                has_nan_lats = True
            frame_lats.append(latitude)

        # interpolate and extrapolate nan values
        frame_longs = np.array(frame_longs)
        if has_nan_longs:
            frame_longs = TimedPoseExtractor._interpolate_nan(frame_longs)

        frame_lats = np.array(frame_lats)
        if has_nan_lats:
            frame_lats = TimedPoseExtractor._interpolate_nan(frame_lats)

        def mse(s: float) -> float:
            inter_longs = np.interp(srt_seconds + s, ad_seconds, ad_longs)
            inter_lats = np.interp(srt_seconds + s, ad_seconds, ad_lats)
            error = np.mean((inter_longs - frame_longs) ** 2 + (inter_lats - frame_lats) ** 2)
            return error

        res = minimize(mse, np.zeros(1), method="Nelder-Mead", options={"disp": False})

        return res.x[0]

    @staticmethod
    def _distribute_timestamps(srt_frames: Sequence[SrtFrame]) -> None:
        """
        Redistribute the timestamps of the given SRT frames to ensure they do not overlap. The objects are edited inplace.
        :param srt_frames: The SRT frames to redistribute.
        """

        timestamp_groups = defaultdict(list)
        for srt_frame in srt_frames:  # group objects by their timestams
            timestamp_groups[srt_frame.timestamp].append(srt_frame)

        # Distribute timestamps for groups with multiple objects
        for timestamp, group in timestamp_groups.items():
            if len(group) > 1:
                # Calculate the time increment between objects
                increment = datetime.timedelta(seconds=1) / len(group)

                # Redistribute timestamps within that second
                for i, obj in enumerate(group):
                    obj.timestamp = timestamp + i * increment


    @staticmethod
    def _srt_to_air_data(srt_frame: SrtFrame, reference_ad_frame: AirDataFrame) -> AirDataFrame:
        """
        Create an AirData frame from an SRT frame with focus on time, position, and orientation data.
        :param srt_frame: The SRT frame to convert.
        :return: An AirData frame.
        """
        conv_frame = AirDataFrame()

        conv_frame.datetime = srt_frame.timestamp
        conv_frame.latitude = srt_frame.latitude
        conv_frame.longitude = srt_frame.longitude
        conv_frame.height_above_takeoff = srt_frame.rel_alt

        # There are differences between the absolute altitude values in SRT and AirData
        # Relative altitudes are comparable. So quick fix based on relative SRT height + initial AirData height
        conv_frame.altitude = reference_ad_frame.altitude - reference_ad_frame.height_above_takeoff + srt_frame.rel_alt
        conv_frame.height_above_ground_at_drone_location = srt_frame.abs_alt
        conv_frame.xSpeed = srt_frame.drone_speedx
        conv_frame.ySpeed = srt_frame.drone_speedy
        conv_frame.zSpeed = srt_frame.drone_speedz
        conv_frame.compass_heading = srt_frame.drone_yaw
        conv_frame.pitch = srt_frame.drone_pitch
        conv_frame.roll = srt_frame.drone_roll
        conv_frame.gimbal_heading = srt_frame.gb_yaw
        conv_frame.gimbal_pitch = srt_frame.gb_pitch
        conv_frame.gimbal_roll = srt_frame.gb_roll

        return conv_frame


    @staticmethod
    def adapt_srt_air_data_frames(srt_frames: Sequence[SrtFrame], ad_frames: Sequence[AirDataFrame], reference_ad_frame: AirDataFrame
                                  ) -> tuple[Sequence[SrtFrame], Sequence[AirDataFrame], Sequence[tuple[datetime, datetime]]]:
        """
        Adjusts the time stamps of the SRT frames by matching them with AirData frames and generates replacements for
        AirData frames from SRT frames if the time between AirData frames exceeds the offset limit.
        :param srt_frames: The SRT frames to adjust.
        :param ad_frames: The AirData frames to adjust.
        :reference_ad_frame: The initial AirData frame to adjust the SRT frames.
        :return: A tuple containing the new SRT frames, the new AirData frames, and the datetime ranges where SRT frames have
        been converted to AirData frames.
        """
        # region adjust SRT frames
        srt_offset = TimedPoseExtractor._get_srt_air_data_offset(srt_frames, ad_frames)
        srt_offset_td = datetime.timedelta(seconds=srt_offset)

        for srt_frame in srt_frames:
            srt_frame.timestamp = srt_frame.timestamp + srt_offset_td
        srt_frame_count = len(srt_frames)
        # endregion

        # region generate AirData replacements
        srt_moving_index = 0
        new_ad_frames = []
        srt_conv_ranges = []

        # get the first frame as a reference point; use slice that excludes this frame in for loop
        prev_ad_frame = ad_frames[0]
        for ad_frame in islice(ad_frames, 1, None):
            offset = ad_frame.time - prev_ad_frame.time

            # if the time difference between the previous and current AirData frame is too high
            if offset > AIR_DATA_MAX_FRAME_OFFSET:

                # move the SRT index until it reaches the first SRT frame after the previous AirData frame
                while srt_moving_index < srt_frame_count and srt_frames[srt_moving_index].timestamp <= prev_ad_frame.datetime:
                    srt_moving_index += 1


                if srt_moving_index < srt_frame_count:  # if index is not out of bounds

                    # get all SRT frame indices for the current timerange
                    first_index = srt_moving_index
                    while srt_moving_index < srt_frame_count and srt_frames[srt_moving_index].timestamp < ad_frame.datetime:
                        srt_moving_index += 1
                    last_index = srt_moving_index

                    # redistribute the SRT frames so there are no duplicate timestamps
                    cur_srt_frames = srt_frames[first_index:last_index]
                    TimedPoseExtractor._distribute_timestamps(cur_srt_frames)

                    # convert all SRT frames between the previous and current AirData frame to AirData frames
                    for srt_frame in cur_srt_frames:
                        cur_conv_frame = TimedPoseExtractor._srt_to_air_data(srt_frame, reference_ad_frame)

                        cur_conv_frame.time = prev_ad_frame.time + (srt_frame.timestamp - prev_ad_frame.datetime).total_seconds() * 1000
                        cur_conv_frame.isPhoto = True
                        cur_conv_frame.isVideo = True

                        new_ad_frames.append(cur_conv_frame)

                    srt_conv_ranges.append((prev_ad_frame.datetime, ad_frame.datetime))

            new_ad_frames.append(ad_frame)
            prev_ad_frame = ad_frame
        # endregion

        # check if airdata is early ending
        if srt_frames[-1].timestamp > ad_frames[-1].datetime:
            # move forward SRT frame indices
            while srt_moving_index < srt_frame_count and srt_frames[srt_moving_index].timestamp <= prev_ad_frame.datetime:
                srt_moving_index += 1

            if 0 < srt_moving_index < srt_frame_count:
                # get all SRT frame indices for the current timerange
                cur_srt_frames = srt_frames[srt_moving_index:-1]
                TimedPoseExtractor._distribute_timestamps(cur_srt_frames)

                # convert all SRT frames between the previous and current AirData frame to AirData frames
                for srt_frame in cur_srt_frames:
                    cur_conv_frame = TimedPoseExtractor._srt_to_air_data(srt_frame, reference_ad_frame)

                    cur_conv_frame.time = prev_ad_frame.time + (
                                srt_frame.timestamp - prev_ad_frame.datetime).total_seconds() * 1000
                    cur_conv_frame.isPhoto = True
                    cur_conv_frame.isVideo = True

                    new_ad_frames.append(cur_conv_frame)

                srt_conv_ranges.append((prev_ad_frame.datetime, ad_frame.datetime))


        return srt_frames, new_ad_frames, srt_conv_ranges

    def _write_image_mask(self, target_folder: str, overwrite_existing: bool) -> str:
        filename = f"mask_{self.camera_name}.png"
        path = os.path.join(target_folder, filename)
        if not os.path.exists(path) or overwrite_existing:
            sr = SensorResolution(self.drone_name, self.camera_name)
            if isinstance(self.frame_accessor, CalibratedVideoFrameAccessor):
                mask = self.frame_accessor.create_distortion_mask(sr.width, sr.height)
                mask[mask < 255] = 0
            else:
                mask = np.full((sr.height, sr.width), 255, dtype=np.uint8)
            cv2.imwrite(path, mask)
        return filename

    @staticmethod
    def _to_index_ranges(timestamp_ranges: Sequence[tuple[datetime.datetime, datetime.datetime]],
                         timestamps: Sequence[datetime.datetime]) -> Sequence[tuple[int, int]]:
        result = []
        for start, end in timestamp_ranges:
            start_index = bisect.bisect_left(timestamps, start)
            end_index = bisect.bisect_right(timestamps, end) - 1
            result.append((start_index, end_index))
        return result

    def extract(
        self,
        target_folder: str,
        air_data_path: str,
        video_paths: Union[str, Sequence[str]],
        srt_data_paths: Union[str, Sequence[str]],
        skip: datetime.timedelta = datetime.timedelta(seconds=0),
        sampling_rate: datetime.timedelta = datetime.timedelta(seconds=0),
        limit: datetime.timedelta = datetime.timedelta(seconds=-1),
        gps_writer: Optional[GpsExifWriter] = None,
        include_gps: bool = True,
        include_time: bool = True,
        mask_images: bool = True,
        origin: Optional[AirDataFrame] = None,
        overwrite_existing: bool = False,
        remove_duplicates: bool = True,
    ) -> Sequence[tuple[int, int]]:
        """
        Method used to extract normalized video frames together with a JSON file describing the relative position between the first frame and the others
        :param target_folder: where normalized frames and JSON file should be written
        :param air_data_path: path to the source air data file describing the meta information of the video frames
        :param video_paths: paths to the video files to extract the video frames from
        :param srt_data_paths: paths to the SRT files to extract the video frames from
        :param skip: Number of frames that should be skipped (no callback called)
        :param sampling_rate: Number of every x-th frame that should be taken (if 0, every frame is used)
        :param limit: Number of frames that should be accessed
        :param gps_writer: Writer used to add GPS information to extracted images
        :param include_gps: Flag that signals if GPS position should be included in created JSON file per frame
        :param include_time: Flag that signals if time should be included in created JSON file per frame
        :param mask_images: Flag that signals if undistorted images should be masked out
        :param origin: Origin used for pose definition (only longitude, latitude and optionally altitude required). If None first position is used.
        :param overwrite_existing: Flag that signals if existing files should be overwritten. default is False.
        :param remove_duplicates: Flag that signals if duplicate frames should be removed. default is True.
        :return:
        """
        _ = skip, sampling_rate, limit, gps_writer

        if isinstance(video_paths, str):
            video_paths = (video_paths,)
        if isinstance(srt_data_paths, str):
            srt_data_paths = (srt_data_paths,)
        if len(video_paths) != len(srt_data_paths):
            raise ValueError("Number of passed video files does not match the number of passed SRT files")

        image_files: list[str] = []
        image_timestamps: list[datetime.datetime] = []

        # match SRT/video frames with AirData frames
        srt_frames, num_videos, frame_to_video = self.get_srt_frames(srt_data_paths)
        start_frame, ad_frames = self.get_air_data_frames(air_data_path, srt_frames[0].timestamp)
        srt_frames, ad_frames, conv_ranges = self.adapt_srt_air_data_frames(srt_frames, ad_frames, start_frame)

        check_duplicate_func = (
            image_equality_check if self.camera_name == Camera.Thermal and remove_duplicates
            else None
        )

        for iv, video_path in enumerate(video_paths):
            srt_video_frames = np.array(srt_frames)[np.array(frame_to_video) == iv]

            # create callback that is called for every frame
            callback = TimedFrameExtractorCallback(
                image_files=image_files,
                image_timestamps=image_timestamps,
                srt_frames=srt_video_frames,
                drone_name=self.drone_name,
                camera_name=self.camera_name,
                target_folder=target_folder,
                overwrite_existing=overwrite_existing,
            )

            self.frame_accessor.access(
                video_path,
                callback,
                check_duplicate_function=check_duplicate_func,
            )

        # interpolate AirData frames to match the image timestamps
        interpolated_frames = AirDataTimeInterpolator(ad_frames)(image_timestamps)

        conv_ranges = self._to_index_ranges(conv_ranges, image_timestamps)

        origin_altitude = 0
        if origin is not None:
            long = origin.longitude
            lat = origin.latitude
            origin_transformed = self.rel_transformer.transform(lat, long)
            origin_altitude = origin.altitude or 0
        else:
            origin = AirDataFrame()
            origin_transformed = [0, 0]

        images: list[dict[str, Any]] = []
        for i, (image_file, image_timestamp) in enumerate(zip(image_files, image_timestamps)):
            frame = interpolated_frames[i]

            frame_altitude = frame.altitude or 0
            frame_coord = self.rel_transformer.transform(frame.latitude, frame.longitude)
            location = [
                frame_coord[0] - origin_transformed[0],
                frame_coord[1] - origin_transformed[1],
                frame_altitude - origin_altitude,
            ]

            rotation = [
                # pitch is rotation around X axis (+90° because per default it faces forward)
                (float(frame.gimbal_pitch) + 90) % 360 if frame.gimbal_pitch is not None else 0.0,
                0,  # roll (Y-axis) is always zero!
                frame.compass_heading if frame.compass_heading is not None else 0.0,  # heading is rotation around Z (up axis)
            ]

            current_dict = {
                "imagefile": image_file,
                "location": location,
                "rotation": rotation,
            }

            if isinstance(self.frame_accessor, CalibratedVideoFrameAccessor):
                current_dict["fovy"] = (self.frame_accessor.undistortion_parameters.fovy,)

            if include_gps:
                current_dict["lat"] = frame.latitude
                current_dict["lng"] = frame.longitude

            if include_time and frame.datetime is not None:
                current_dict["timestamp"] = frame.datetime.isoformat()

            images.append(current_dict)

        res = dict()
        res["images"] = images
        res["origin"] = {"latitude": origin.latitude, "longitude": origin.longitude, "altitude": origin.altitude}
        res["drone"] = ("Unknown" if self.drone_name is None else Drone.product_name(self.drone_name))
        res["camera"] = ("Unknown" if self.camera_name is None else Camera.fullname(self.camera_name))
        res["samplingRate"] = 0

        if mask_images and isinstance(self.frame_accessor, CalibratedVideoFrameAccessor):
            res["mask"] = self._write_image_mask(target_folder, overwrite_existing)

        # store the result in a json file
        res_path = os.path.join(target_folder, "poses.json")
        with open(res_path, "w+", encoding="UTF-8") as jf:
            json.dump(res, jf)

        return conv_ranges
