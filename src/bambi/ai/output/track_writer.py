import abc
import os.path
from typing import List

from bambi.ai.domain.Track import Track


class TrackWriter(abc.ABC):
    """
    Abstract class used to write tracks
    """
    def write_tracks(self, input_path: str, output_path: str, labels: List[str], detections: List[Track]):
        """
        :param input_path: Source of data
        :param output_path: Target to write detections
        :param labels: To be expected
        :param detections: To be written
        :return: None
        """
        for i, t in enumerate(detections):
            self.write_track(input_path, os.path.join(output_path, f"{i}"), labels, t)


    @abc.abstractmethod
    def write_track(self, input_path: str, output_path: str, labels: List[str], detection: Track):
        """
        :param input_path: Source of data
        :param output_path: Target to write detections
        :param labels: To be expected
        :param detection: To be written
        :return: None
        """
        pass
