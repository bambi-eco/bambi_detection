import random
from typing import List, Any, Optional

import cv2
import numpy.typing as npt

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.video.video_frame_accessor import VideoFrameAccessor
from bambi.video.video_writer import AbstractVideoWriter, OpenCvVideoWriter


class BoundingBoxVisualizer:
    """
    Class allowing to visualize bounding boxes on a video
    """

    def __init__(self, video_writer: AbstractVideoWriter = OpenCvVideoWriter()) -> None:
        self.__writer = video_writer

    def visualize(self, input_path: str, output_path: str, bbs: List[BoundingBox]) -> None:
        """
        Help method allowing to visualize bounding boxes in videos
        :param input_path: input video
        :param output_path: output video
        :param bbs: to be visualized
        :return: Nothing
        """
        index_bbs = {}

        for bb in bbs:
            idx = bb.idx
            if index_bbs.get(idx) is None:
                index_bbs[idx] = []
            index_bbs[idx].append(bb)

        accessor = VideoFrameAccessor()

        def process(idx: int, frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
            bbs = index_bbs.get(idx)
            self.visualize_frame(frame, bbs)
            return frame

        self.__writer.write(target_path=output_path, frames=accessor.access_yield(input_path), callback=process)

    def visualize_frame(self, frame: npt.NDArray[Any], bbs: Optional[List[BoundingBox]]) -> npt.NDArray[Any]:
        """
        Help method allowing to draw bounding boxes on a frame
        :param frame: canvas to be painted
        :param bbs: to be drawn
        :return: Manipulated frame
        """
        if bbs is None:
            return frame
        for bb in bbs:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (b, g, r)
            start_point = (int(bb.start_x), int(bb.start_y))
            end_point = (int(bb.end_x), int(bb.end_y))
            frame = cv2.rectangle(frame, start_point, end_point, color, 2)
        return frame