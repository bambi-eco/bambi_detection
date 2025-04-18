import abc
import os
from typing import List, Tuple, Any, Generator, Callable, Optional

import cv2

from bambi.ai.domain.BoundingBox import BoundingBox
import numpy.typing as npt

from bambi.video.video_frame_accessor import VideoFrameAccessor


class BoundingBoxReader(abc.ABC):
    """
    Abstract class for reading bounding boxes from a file representation
    """

    @abc.abstractmethod
    def read_box(self, input_path: str, index: int = 0) -> List[BoundingBox]:
        pass

    @abc.abstractmethod
    def read_boxes(self, input_paths: List[str]) -> Generator[Tuple[int, List[BoundingBox]], None, None]:
        pass

def read_with_frames(video_accessor: Generator[Tuple[int, npt.NDArray[Any]], None, None], reader: Generator[Tuple[int, List[BoundingBox]], None, None]) -> Generator[Tuple[int, npt.NDArray[Any], List[BoundingBox]], None, None]:
    """
    Help method allowing to combine video frames and bounding boxes
    :param video_accessor: e.g. VideoFrameAccessor#access_yield()
    :param reader: E.g. BoundingBoxReader#read_boxes()
    :return:
    """
    try:
        idx, frame = next(video_accessor)
        for box_idx, boxes in reader:
            while idx < box_idx:
                idx, frame = next(video_accessor)
            if idx == box_idx:
                yield idx, frame, boxes
    except StopIteration as e:
        # Nothing to do
        pass

def read_with_frames_pairwise(frame_paths: List[str], boundingbox_paths: List[str], read_box_function: Callable[[str], List[BoundingBox]]) -> Generator[Tuple[int, npt.NDArray[Any], List[BoundingBox]], None, None]:
    """
    Help method for reading pairs of video frames as images and boundingboxs
    (Frames and bounding boxes must be named after the frame's index)
    :param frame_paths: paths of video frames
    :param boundingbox_paths: paths of bounding boxes
    :param read_box_function: function used to read a bounding box file and create bounding boxes
    :return: Generator of frames and the associated bounding boxes
    """
    sort_function = lambda v: int(os.path.splitext(os.path.basename(v))[0])
    boundingbox_paths = sorted(boundingbox_paths, key=sort_function)

    keyed_paths = {}
    for path in frame_paths:
        idx = int(os.path.splitext(os.path.basename(path))[0])
        keyed_paths[idx] = path

    for boundingbox_path in boundingbox_paths:
        idx = int(os.path.splitext(os.path.basename(boundingbox_path))[0])
        frame = cv2.imread(keyed_paths[idx])
        boxes = read_box_function(boundingbox_path)
        yield idx, frame, boxes



