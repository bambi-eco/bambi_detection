import math
from typing import List, Tuple, Optional, Any, Iterable

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.util.basic_tracker import BasicTracker
import numpy.typing as npt

class BoundingBoxTracker(BasicTracker):
    def __init__(self, max_idx_offset: int = 5, use_center_distance: bool = False, factor: Optional[float] = None):
        """
        :param max_idx_offset: max_idx_offset to decide if track is considered for check
        :param use_center_distance: Flag if the distance between the bounding box centers should be used to determine nearest track
        :param factor: minimum overlapping factor (only used if use_center_distance == False) or aximal distance between two bounding boxes to be considered as associated (only used if use_center_distance == True)
        """
        super().__init__(max_idx_offset, use_center_distance, factor)
    def track(self, detections: Iterable[Tuple[int, npt.NDArray[Any], List[BoundingBox]]]) -> List[Track]:
        """
        Method for tracking bounding boxes
        :param detections: detected bounding boxes, needed for track creation
        :return: Found tracks
        """
        tracks = []
        for idx, _, bbs in detections:
            for bb in bbs:
                new_track = True
                if len(tracks) == 0:
                    t = Track()
                else:
                    t = self.find_nearest_track(tracks, bb)
                    if t is None:
                        t = Track()
                    else:
                        new_track = False
                t.add_bounding_box(bb)
                if new_track:
                    tracks.append(t)
        return tracks

    def find_nearest_track(self, tracks: Iterable[Track], bb: BoundingBox) -> Optional[Track]:
        """
        Returns the nearest track for the given bounding box
        :param tracks: to check
        :param bb: to check
        :return: Nearest track or None if no track matches the condition
        """

        tracks_to_check = (track for track in tracks if abs(bb.idx - track.get_last_idx()) <= self._max_idx_offset)

        if self._use_center_distance:
            best_value = None
        else:
            best_value = 0.0
        current_track = None
        for track in tracks_to_check:
            if self._use_center_distance:
                distance = track.bounding_boxes[track.get_last_idx()].distance(bb)
                if distance < self._factor and (best_value is None or distance <= best_value):
                    best_value = distance
                    current_track = track
            else:
                current_overlap = track.overlap_with_last_bb(bb)
                if current_overlap >= self._factor and current_overlap > best_value:
                    best_value = current_overlap
                    current_track = track

        return current_track