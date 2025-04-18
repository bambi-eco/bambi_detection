import random
from typing import List, Any

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.visualization.line_style import LineStyle
from bambi.ai.visualization.visualization_util import draw_rect
from bambi.video.video_frame_accessor import VideoFrameAccessor
from bambi.video.video_writer import AbstractVideoWriter, OpenCvVideoWriter
import numpy.typing as npt

class TrackVisualizer:
    """
    Class allowing to visualize tracks on a video
    """
    def __init__(self, video_writer: AbstractVideoWriter = OpenCvVideoWriter(), line_thickness: int = 2, line_gap: int = 20) -> None:
        self.__writer = video_writer
        self.__line_thickness = line_thickness
        self.__line_gap = line_gap

    def visualize(self, input_path: str, output_path: str, tracks: List[Track]) -> None:
        """
        :param input_path: Path for the input video
        :param output_path: Path for the output video
        :param tracks: to be visualized
        :return: None
        """
        bounding_boxes = {}

        for track in tracks:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            for key in sorted(track.bounding_boxes.keys()):
                bb = track.bounding_boxes.get(key)
                if bounding_boxes.get(key) is None:
                    bounding_boxes[key] = []
                bounding_boxes[key].append({
                    "bb" : bb,
                    "color" : (b, g, r)
                })
        accessor = VideoFrameAccessor()

        def process(idx: int, frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
            bbs = bounding_boxes.get(idx)
            if bbs is not None:
                for bb_dict in bbs:
                    color = bb_dict["color"]
                    box: BoundingBox = bb_dict["bb"]
                    start_point = (int(box.start_x), int(box.start_y))
                    end_point = (int(box.end_x), int(box.end_y))
                    style = LineStyle.DASHED if box.is_probagated else LineStyle.SOLID
                    draw_rect(frame, start_point, end_point, color, self.__line_thickness, style, self.__line_gap)
            return frame

        self.__writer.write(target_path=output_path, frames=accessor.access_yield(input_path), callback=process)
