import abc
from typing import List

from bambi.ai.domain.Track import Track
from bambi.ai.inference import Inference
from bambi.ai.output.track_writer import TrackWriter

class Tracking(Inference, abc.ABC):
    """
    Abstract class representing a AI model for tracking
    """
    def __init__(self, track_writer: TrackWriter, labels: List[str]):
        self._labels = labels
        self._track_writer = track_writer

    @abc.abstractmethod
    def track(self, input_path: str) -> List[Track]:
        """
        Method for applying the model to the given input
        :param input_path: Input that should be analysed with AI inference
        :return: Generator with frame indices and the list of detected bounding boxes
        """
        pass

    def apply(self, input_path: str, output_path: str) -> None:
        self._track_writer.write_tracks(input_path, output_path, self._labels, self.track(input_path))

    def get_labels(self) -> List[str]:
        """
        :return: Labels used by the detection model
        """
        return self._labels

