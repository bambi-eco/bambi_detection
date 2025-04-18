# pylint: disable=R0201
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from bambi.video.video_domain import UndistortionParameters, VideoInput
from bambi.video.video_frame_accessor import (
    MultiVideoFrameAccessor,
    VideoFrameAccessor,
)


class CalibratedVideoFrameAccessor(VideoFrameAccessor):
    """
    Class that allows to access undistorted video frames
    """

    def __init__(
        self,
        calibration_res: Dict[str, Any],
        _new_size: Optional[Tuple[int, int]] = None,
        _new_camera_matrix: Optional[npt.NDArray[Any]] = None,
    ):
        self.calibration_res = calibration_res
        self.undistortion_parameters = UndistortionParameters(
            _new_camera_matrix, _new_size
        )

    def access_yield(
        self,
        video_path: str,
        skip: int = 0,
        sampling_rate: int = 0,
        limit: Optional[int] = None,
        check_duplicate_function: Optional[
            Callable[[npt.NDArray[Any], npt.NDArray[Any]], bool]
        ] = None,
        read_grayscale: bool = False,
    ) -> Generator[Tuple[int, npt.NDArray[Any]], None, None]:
        """
        Method that allows to access video frames in a generator like way
        :param video_path: path to the video file
        :param skip: Number of frames that should be skipped (no callback called)
        :param sampling_rate: Number of every x-th frame that should be taken (if 0, every frame is used)
        :param limit: Number of frames that should be accessed
        :param check_duplicate_function: Method allowing to check if two frames are duplicated (still frames). Duplicated frames are ignored if parameter is not None
        :param read_grayscale: Flag if video should be read grayscale or bgr
        :return: None
        """

        for (idx, frame) in super().access_yield(
            video_path,
            skip,
            sampling_rate,
            limit,
            check_duplicate_function,
            read_grayscale=read_grayscale,
        ):
            yield idx, self.undistort(frame)

    def access(
        self,
        video_path: str,
        callback: Callable[[int, npt.NDArray[Any]], bool],
        skip: int = 0,
        sampling_rate: int = 0,
        limit: Optional[int] = None,
        check_duplicate_function: Optional[
            Callable[[npt.NDArray[Any], npt.NDArray[Any]], bool]
        ] = None,
        read_grayscale: bool = False,
    ) -> UndistortionParameters:
        """
        Method that allows to access video frames, calling a given callback function for every frame
        :param video_path: path to the video file
        :param callback: callback that is executed for every individual frame
        :param calibration_res: Calibration result used for the undistortion process
        :param skip: Number of frames that should be skipped (no callback called)
        :param sampling_rate: Number of every x-th frame that should be taken (if 0, every frame is used)
        :param limit: Number of frames that should be accessed
        :param check_duplicate_function: Method allowing to check if two frames are duplicated (still frames). Duplicated frames are ignored if parameter is not None
        :param read_grayscale: Flag if video should be read grayscale or bgr
        :return: Calibration parameters used to undistort image
        """
        for (idx, frame) in self.access_yield(
            video_path,
            skip,
            sampling_rate,
            limit,
            check_duplicate_function,
            read_grayscale=read_grayscale,
        ):
            if not callback(idx, frame):
                break

        return self.undistortion_parameters

    def prepare_undistort(
        self,
        img_size: Tuple[int, int],
        alpha: Optional[float] = 0.5,
        center_principal_point: Optional[bool] = True,
        force_same_fov: Optional[bool] = True,
    ) -> UndistortionParameters:
        """Method to initialize the undistortion maps which are used in undistort

        :param img_size: The image size of the input image
        :param calibration: A dictionary storing the calibration (matrix and distortion coefficients)
        :param new_size: The new image size after distortion
        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image). see @https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1
        :param center_principal_point: Optional flag that indicates whether in the new camera intrinsic matrix the principal point should be at the image center or not. By default, the principal point is chosen to best fit a subset of the source image (determined by alpha) to the corrected image.
        :param force_same_fov: Enforce the same fov in x and y (fovx and fovy).
        """
        w, h = img_size
        if self.undistortion_parameters.new_size is None:
            wh = min(h, w)
            self.undistortion_parameters.new_size = (wh, wh)
            # print("WARNING: No new size defined, using square image!")
        mtx = np.asarray(self.calibration_res["mtx"])
        dist = np.asarray(self.calibration_res["dist"])
        if self.undistortion_parameters.new_camera_matrix is None:
            new_cameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx,
                dist,
                (w, h),
                alpha,
                self.undistortion_parameters.new_size,
                centerPrincipalPoint=center_principal_point,
            )
            if force_same_fov:
                assert (
                    self.undistortion_parameters.new_size[0]
                    == self.undistortion_parameters.new_size[1]
                ), "The new image size must be square to force the same fov in x and y!"
                fxy = max(new_cameramtx[0, 0], new_cameramtx[1, 1])
                new_cameramtx[0, 0] = fxy
                new_cameramtx[1, 1] = fxy

            self.undistortion_parameters.new_camera_matrix = new_cameramtx
            # print("WARNING: No new camera matrix defined, using default!")

        # prepare undistortion maps to undistort images
        mapx, mapy = cv2.initUndistortRectifyMap(
            mtx,
            dist,
            None,
            self.undistortion_parameters.new_camera_matrix,
            self.undistortion_parameters.new_size,
            5,
        )

        # store
        self.undistortion_parameters.mapx = mapx
        self.undistortion_parameters.mapy = mapy
        self.undistortion_parameters.is_initialized = True
        return self.undistortion_parameters

    def undistort(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Method for undistorting a single video frame
        :param img: to be undistorted
        :return: undistorted frame
        """

        if not self.undistortion_parameters.is_initialized:
            h, w = img.shape[:2]
            self.prepare_undistort((w, h))

        # undistort image
        dst = cv2.remap(
            img,
            self.undistortion_parameters.mapx,
            self.undistortion_parameters.mapy,
            cv2.INTER_LINEAR,
        )

        return dst

    def create_distortion_mask(self, width: int, height: int) -> npt.NDArray[Any]:
        """
        Method for creating a mask showing the applied distortion
        :param width: width of the mask
        :param height: height of the mask
        :return:
        """
        img = np.full((height, width), 255, dtype=np.uint8)
        return self.undistort(img)


class MultiCalibratedVideoFrameAccessor(MultiVideoFrameAccessor):
    """
    Class allowing to access multiple videos, with calibrated (undistorted) video frames
    """

    def _get_generators(self, videos: List[VideoInput]):
        """
        Method for defining the iterators used to access the individual video frames
        :param videos: to be accessed
        :return: List of generator
        """
        generators = []
        for video in videos:
            path = video.video_path
            limit = video.limit
            skip = video.skip
            sampling_rate = video.sampling_rate
            calibration_res = video.calibration_res
            read_grayscale = video.read_grayscale
            accessor = CalibratedVideoFrameAccessor(calibration_res)
            generators.append(
                accessor.access_yield(
                    path,
                    skip,
                    sampling_rate,
                    limit,
                    read_grayscale=read_grayscale,
                )
            )
        return generators
