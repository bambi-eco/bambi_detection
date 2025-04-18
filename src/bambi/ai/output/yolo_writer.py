import os.path
from typing import List, Tuple, Any, Union

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.output.boundingbox_writer import BoundingBoxWriter
import numpy.typing as npt

class YoloWriter(BoundingBoxWriter):
    """
    Class allowing to write bounding boxes as yolo format
    """
    def write_box(self, output_path: str, labels: List[str], detections: Tuple[int, Union[npt.NDArray[Any], Tuple[int, int]], List[BoundingBox]]):
        if not output_path.endswith(".txt"):
            output_path += ".txt"
        with open(output_path, "w") as file:
            frame = detections[1]
            if type(frame) is tuple:
                width = frame[0]
                height = frame[1]
            else:
                width = frame.shape[1]
                height = frame.shape[0]
            for bb in detections[2]:
                # cut bb to image size
                if bb.start_x < 0:
                    bb.start_x = 0
                if bb.start_y < 0:
                    bb.start_y = 0
                if bb.end_x > width:
                    bb.end_x = width
                if bb.end_y > height:
                    bb.end_y = height

                # if bb area is smaller than 20px, skip it
                if bb.get_width() * bb.get_height() < 20:
                    continue

                center = bb.get_center_on_image()
                rel_center_x = center[0] / width
                rel_center_y = center[1] / height
                bb_width = bb.get_width() / width
                bb_height = bb.get_height() / height

                if bb.label is None:
                    raise Exception("BoundingBox Label must not be None")

                file.write(f"{labels.index(bb.label)} {rel_center_x} {rel_center_y} {bb_width} {bb_height}\n")