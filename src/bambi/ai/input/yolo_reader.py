import os
from typing import List, Generator, Tuple

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.input.boundingbox_reader import BoundingBoxReader


class YoloReader(BoundingBoxReader):
    """
    Bounding box reader implementation for the yolo format
    """
    def __init__(self, video_width: int, video_height: int, labels: List[str]):
        self._video_width = video_width
        self._video_height = video_height
        self._labels = labels

    def read_box(self, input_path: str, index: int = 0) -> List[BoundingBox]:
        res = []
        with open(input_path, "r") as file:
            for line in file.readlines():
                if len(line) == 0:
                    continue
                label = None
                center_x = None
                center_y = None
                bb_width = None
                bb_height = None
                for idx, split in enumerate(line.split(" ")):
                    if idx == 0:
                        label = self._labels[int(split)]
                    elif idx == 1:
                        center_x = float(split) * self._video_width
                    elif idx == 2:
                        center_y = float(split) * self._video_height
                    elif idx == 3:
                        bb_width = float(split) * self._video_width
                    elif idx == 4:
                        bb_height = float(split) * self._video_height

                half_width = bb_width / 2
                half_height = bb_height / 2
                start_x = center_x - half_width
                start_y = center_y - half_height
                end_x = center_x + half_width
                end_y = center_y + half_height
                res.append(BoundingBox(index, start_x, start_y, end_x, end_y, label))

        return res

    def read_boxes(self, input_paths: List[str]) -> Generator[Tuple[int, List[BoundingBox]], None, None]:
        for input_path in input_paths:
            idx = int(os.path.splitext(os.path.basename(input_path))[0])
            bbs = self.read_box(input_path, idx)
            yield idx, bbs
