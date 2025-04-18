import datetime
import json
import math
import os.path
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
import numpy.typing as npt
import pyproj
from attr import define, field
import cv2
from pyproj import CRS, Transformer

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_parser import AirDataParser, AirDataParserInterface
from bambi.domain.camera import Camera
from bambi.domain.drone import Drone
from bambi.geo.gps_exif_writer import GpsExifWriter
from bambi.video.calibrated_video_frame_accessor import (
    CalibratedVideoFrameAccessor,
)


class PoseExtractor:
    """
    Class that allows to extract normalized video frames together with a JSON file describing the relative position between the first frame and the others
    """

    def __init__(
        self,
        calibrated_frame_accessor: CalibratedVideoFrameAccessor,
        parser: AirDataParserInterface,
        rel_transformer: Transformer = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(32633)
        ),
        drone_name: Optional[Drone] = None,
        camera_name: Optional[Camera] = None,
    ):
        """
        :param calibrated_frame_accessor: Object used to access the individual, normalized video frames
        :param parser: Used to access the AirData file describing the meta information of the video frames
        :param rel_transformer: Used to calculate the relative position between from the reference point per frame
        :param drone_name: Drone used to create video
        :param camera_name: Camera used to create video
        """
        self.calibrated_frame_accessor = calibrated_frame_accessor
        self.parser = parser
        self.rel_transformer = rel_transformer
        self.drone_name = drone_name
        self.camera_name = camera_name

    def extract(
        self,
        target_folder: str,
        video_path: str,
        air_data_path: str,
        fps: float,
        video_time: Optional[datetime.datetime] = None,
        skip: int = 0,
        sampling_rate: int = 0,
        limit: Optional[int] = None,
        apply_correction: bool = False,
        gps_writer: Optional[GpsExifWriter] = None,
        include_gps: bool = False,
        mask_images: bool = False,
        origin: Optional[AirDataFrame] = None,
    ) -> None:
        """
        Method used to extract normalized video frames together with a JSON file describing the relative position between the first frame and the others
        :param target_folder: where normalized frames and JSON file should be written
        :param video_path: path to the source video
        :param air_data_path: path to the source air data file describing the meta information of the video frames
        :param fps: Frames per second of the video (required to associate the AirData frames with the video frames)
        :param video_time: Creation time of the video (in UTC timezone)
        :param skip: Number of frames that should be skipped (no callback called)
        :param sampling_rate: Number of every x-th frame that should be taken (if 0, every frame is used)
        :param limit: Number of frames that should be accessed
        :param apply_correction: Whether to apply a transformation based rotation correction
        :param gps_writer: Writer used to add GPS information to extracted images
        :param include_gps: Flag that signals if GPS position should be included in created JSON file per frame
        :param mask_images: Flag that signals if undistorted images should be masked out
        :param origin: Origin used for pose definition (only longitude, latitude and optionally altitude required). If None first position is used.
        :return:
        """
        # array holding the index and the current frame_time
        current_air_data_frame = [-1, -1]

        images: List[Dict[str, Any]] = []
        res: Dict[str, Any] = {"images": images}

        # In SRT there is meta information for every frame of the video
        # But in AirData we have meta information in 100ms steps and data before the video recording was started,
        # so we have to always find the correct frame and offset!
        is_airdata = isinstance(self.parser, AirDataParser)
        offset_frame_time = 0
        if (
            is_airdata
            and video_time is not None
            and isinstance(self.parser, AirDataParser)
        ):
            offset_frame_time = self.parser.get_file_video_offset(air_data_path, video_time)

        parse_yield = self.parser.parse_yield(air_data_path)
        frame = next(parse_yield)
        if frame is None:
            raise Exception(f"Could not extract initial frame from AirData file!")

        callback = PoseExtractorCallback(
            images=images,
            frame=frame,
            rel_transformer=self.rel_transformer,
            parse_yield=parse_yield,
            target_folder=target_folder,
            fpms=1000 / fps,  # duration of one frame in ms
            offset_frame_time=offset_frame_time,
            is_airdata=is_airdata,
            fovy_callback=lambda: self.calibrated_frame_accessor.undistortion_parameters.fovy,
            apply_correction=apply_correction,
            gps_writer=gps_writer,
            drone_name=self.drone_name,
            camera_name=self.camera_name,
            reference_frame=origin,
            include_gps=include_gps,
            mask_images=mask_images,
        )

        self.calibrated_frame_accessor.access(
            video_path, callback, skip, sampling_rate, limit
        )

        # add meta information to the JSON file
        if callback.reference_frame is None:
            raise Exception(f"Could not extract initial frame from AirData file!")

        res["origin"] = dict(
            latitude=callback.reference_frame.latitude,
            longitude=callback.reference_frame.longitude,
            altitude=callback.reference_frame.altitude,
        )
        res["drone"] = (
            "Unknown"
            if self.drone_name is None
            else Drone.product_name(self.drone_name)
        )

        res["samplingRate"] = sampling_rate

        # store the result in a json file
        res_path = os.path.join(target_folder, "poses.json")
        with open(res_path, "w", encoding="UTF-8") as file:
            json.dump(res, file)


@define
class PoseExtractorCallback:
    """Callback class that is used to extract normalized video frames together with a JSON file describing the relative
    position between the first frame and the others"""

    images: List[Dict[str, Any]]
    frame: AirDataFrame
    offset_frame_time: int
    is_airdata: bool
    rel_transformer: Transformer
    target_folder: str
    fpms: float
    fovy_callback: Callable[[], float]
    parse_yield: Generator[AirDataFrame, None, None]
    apply_correction: bool
    drone_name: Optional[Drone]
    camera_name: Optional[Camera]
    reference_frame: Optional[AirDataFrame] = field(default=None)
    gps_writer: Optional[GpsExifWriter] = None
    include_gps: bool = False
    mask_images: bool = False

    _reference_transformed: Optional[tuple[float, float]] = None

    def __call__(self, idx: int, img: npt.NDArray[Any]) -> bool:
        previous_frame = self.frame
        if self.is_airdata:
            while (self.frame.time - self.offset_frame_time) <= idx * self.fpms:
                previous_frame = self.frame
                self.frame = next(self.parse_yield)
        else:
            while True:
                previous_frame = next(self.parse_yield)
                if previous_frame.id == idx:
                    break

        if self.frame is None:
            raise Exception(
                f"Could not extract frame from AirData file (associated to video frame {idx})"
            )
        if self.is_airdata and self.frame.isVideo == 0:
            print(
                f"AirData frame (row {self.frame.id}) is not associated with a video (isVideo is 0). Current video frame id {idx}"
            )
            return False

        # for storage and so on we use the previous frame
        frame = previous_frame
        frame_idx = frame.id

        if self.reference_frame is None:
            self.reference_frame = frame  # init the reference frame on first use!
        reference = self.reference_frame

        if self._reference_transformed is None:
            long = reference.longitude
            lat = reference.latitude
            self._reference_transformed = self.rel_transformer.transform(lat, long)

        frame_long = frame.longitude
        frame_lat = frame.latitude

        correction_angle = 0
        if self.apply_correction:
            target_crs = self.rel_transformer.target_crs
            cor_transformer = Transformer.from_crs("EPSG:4236", target_crs)
            geod = pyproj.Geod(ellps="WGS84")  # assume source coords use WGS84

            north_point_long, north_point_lat, _ = geod.fwd(
                frame_long, frame_lat, frame.gimbal_heading or 0.0, 1
            )
            north_point_long_target, north_point_lat_target = cor_transformer.transform(
                north_point_long, north_point_lat
            )

            long_target, lat_target = self.rel_transformer.transform(
                frame_long, frame_lat
            )

            north_fwd_azimuth, _, north_distance = geod.inv(
                long_target, lat_target, north_point_long_target, north_point_lat_target
            )
            correction_angle = (
                90
                + math.atan2(
                    lat_target - north_point_lat_target,
                    long_target - north_point_long_target,
                )
                * 180
                / math.pi
            )

        frame_altitude = frame.altitude or 0
        reference_altitude = reference.altitude or 0

        height_diff = frame_altitude - reference_altitude

        frame_coord = self.rel_transformer.transform(frame_lat, frame_long)
        location = [
            frame_coord[0] - self._reference_transformed[0],
            frame_coord[1] - self._reference_transformed[1],
            height_diff,
        ]

        filename = f"{idx}-{frame_idx}.png"
        path = os.path.join(self.target_folder, filename)
        if self.mask_images:
            # TODO calculate mask from distortion parameters (currently there is a bug)
            mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
            mask[np.all(img == (0, 0, 0), axis=-1)] = 0
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            img[:, :, 3] = mask
        cv2.imwrite(path, img)
        if self.gps_writer is not None:
            drone_name = (
                "Unknown"
                if self.drone_name is None
                else Drone.product_name(self.drone_name)
            )
            camera_name = (
                "Unknown" if self.camera_name is None else str(self.camera_name)
            )
            self.gps_writer.write_gps(
                path, frame_long, frame_lat, frame_altitude, drone_name, camera_name
            )

        current_dict: Dict[str, Any] = {
            "imagefile": filename,
            "location": location,
            "rotation": [
                float(frame.gimbal_pitch) + 90 if frame.gimbal_pitch is not None else 0.0,
                # pitch is rotation around X axis (+90° because per default it faces forward)
                0,  # roll (Y-axis) is always zero!
                (frame.compass_heading + correction_angle)
                if frame.compass_heading is not None
                else 0.0,  # heading is rotation around Z (up axis)
            ],
            "fovy": self.fovy_callback(),
            "timestamp": frame.datetime.isoformat(),
        }

        if self.include_gps:
            current_dict["lat"] = frame_lat
            current_dict["lng"] = frame_long

        assert self.fovy_callback() > 0, f"fovy is {self.fovy_callback()}!"

        self.images.append(current_dict)

        return True
