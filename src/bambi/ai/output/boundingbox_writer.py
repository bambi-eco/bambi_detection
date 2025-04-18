import abc
import os
from typing import Tuple, List, Any
import numpy.typing as npt
from bambi.ai.domain.BoundingBox import BoundingBox


class BoundingBoxWriter(abc.ABC):
    """
    Abstract class used to write bounding boxes
    """
    def write_boxes(self, output_path: str, labels: List[str], detections: List[Tuple[int, npt.NDArray[Any], List[BoundingBox]]]) -> None:
        """
        :param input_path: Source of data
        :param output_path: Target to write detections
        :param labels: To be expected
        :param detections: To be written
        :return: None
        """
        for t in detections:
            self.write_box(os.path.join(output_path, f"{t[0]}"), labels, t)

    @abc.abstractmethod
    def write_box(self, output_path: str, labels: List[str], detections: Tuple[int, npt.NDArray[Any], List[BoundingBox]]):
        """
        :param input_path: Source of data
        :param output_path: Target to write detections
        :param labels: To be expected
        :param detections: To be written
        :return: None
        """
        pass

