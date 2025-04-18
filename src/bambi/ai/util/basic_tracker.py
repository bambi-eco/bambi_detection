import abc
from typing import Optional, Tuple, Any, List, Iterable

import numpy.typing as npt
from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track


class BasicTracker(abc.ABC):
    def __init__(self, max_idx_offset: int = 5, use_center_distance: bool = False, factor: Optional[float] = None):
        """
        :param max_idx_offset: max_idx_offset to decide if track is considered for check
        :param use_center_distance: Flag if the distance between the bounding box centers should be used to determine nearest track
        :param factor: minimum overlapping factor (only used if use_center_distance == False) or aximal distance between two bounding boxes to be considered as associated (only used if use_center_distance == True)
        """
        self._max_idx_offset = max_idx_offset
        self._use_center_distance = use_center_distance
        if factor is None:
            self._factor = 0.6 if use_center_distance else 5
        else:
            self._factor = factor

    @abc.abstractmethod
    def track(self, detections: Iterable[Tuple[int, npt.NDArray[Any], List[BoundingBox]]]) -> List[Track]:
        """
        Method allowing to create tracks based on detections
        :param detections: detections of a detection model
        :return: Tracks
        """
        pass
