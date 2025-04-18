import abc
from typing import List, Generator, Tuple, Any

import numpy.typing as npt

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.inference import Inference
from bambi.ai.output.boundingbox_writer import BoundingBoxWriter


class Detection(Inference, abc.ABC):
    """
    Abstract class representing a AI model for bounding box based detections
    """
    def __init__(self, writer: BoundingBoxWriter, labels: List[str]):
        self._detection_writer = writer
        self._labels = labels

    @abc.abstractmethod
    def detect(self, input_path: str) -> Generator[Tuple[int, npt.NDArray[Any], List[BoundingBox]], None, None]:
        """
        Method for applying the model to the given input
        :param input_path: Input that should be analysed with AI inference
        :return: Generator with frame indices and the list of detected bounding boxes
        """
        pass

    @abc.abstractmethod
    def detect_frame(self, idx: int, frame: npt.NDArray[Any]) -> List[BoundingBox]:
        """
        Method for applying the model to a single frame
        :param idx: Index of the frame
        :param frame: used for detection
        :return: found bounding boxes
        """
        pass

    def get_labels(self) -> List[str]:
        """
        :return: Labels used by the detection model
        """
        return self._labels

    def apply(self, input_path: str, output_path: str) -> None:
        self._detection_writer.write_boxes(input_path, output_path, self._labels, [bb for bb in self.detect(input_path)])
