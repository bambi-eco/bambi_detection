import math
from typing import Dict, Optional, List

from bambi.ai.domain.BoundingBox import BoundingBox


class Track:
    """
    Class representing a track of labels accross a video
    """
    def __init__(self):
        self.bounding_boxes: Dict[int, BoundingBox] = {}
        self.__last_idx = -1
        self.track_id: Optional[int] = None

    def get_last_idx(self) -> int:
        """
        :return: Last frame idx of the latest bounding box in this bounding box
        """
        return self.__last_idx

    def get_class(self, min_confidence: float = 0.7, calculate_weighted: bool = True) -> Optional[str]:
        """
        Method for getting the class of this track based on the individual bounding boxes
        :param min_confidence: Min confidence of a bounding box to be considered for the track's class
        :param calculate_weighted: Either calculate the class based on the weighted confidences or based on the number of occurences
        :return: Most likely class
        """
        classes = {}
        for bb in self.bounding_boxes.values():
            if bb.propability is not None and bb.propability >= min_confidence:
                if classes.get(bb.label) is None:
                    classes[bb.label] = []
                classes[bb.label].append(bb.propability)

        most_confident_class = None
        if calculate_weighted:
            highest_confidence = 0
            for key, value in classes.items():
                confidence = sum(value) / len(value)
                if highest_confidence < confidence:
                    highest_confidence = confidence
                    most_confident_class = key
        else:
            highest_confidence = 0
            for key, value in classes.items():
                confidence = len(value)
                if highest_confidence < confidence:
                    highest_confidence = confidence
                    most_confident_class = key
        return most_confident_class

    def get_confidence(self, min_confidence: float = 0.7, calculate_weighted: bool = True):
        """
        :return: Get confidence for track based on bounding box confidences based on the most probable class
        :param min_confidence: Min confidence of a bounding box to be considered for the track's class
        :param calculate_weighted: Either calculate the class based on the weighted confidences or based on the number of occurences
        """
        most_confident_class = self.get_class(min_confidence, calculate_weighted)
        confidences = [bb.propability for bb in self.bounding_boxes.values() if bb.propability is not None and bb.label == most_confident_class]
        if len(confidences) == 0:
            return 0.0
        return sum(confidences) / len(confidences)


    def add_bounding_box(self, bb: BoundingBox) -> None:
        """
        Method allowing to add a bounding box to this track
        :param bb: to be added
        :return:
        """
        if bb.idx < 0:
            raise Exception(f"Illegal bounding box index {bb.idx}")

        self.bounding_boxes[bb.idx] = bb
        if bb.idx > self.__last_idx:
            self.__last_idx = bb.idx

    def belongs_to_track(self, bb: BoundingBox, max_idx_offset: int = 5, factor: float = 0.8) -> bool:
        """
        Method allowing to check if a bounding box belongs to this track based on overlapping area of the given bounding box and the latest bounding box in the track
        :param bb: to be checked
        :param max_idx_offset: max supported frame index offset between current bb and latest bb of track
        :param factor: overlapping factor used to check if the two bounding boxes are overlapping
        :return: True if bb belongs to track, else False
        """
        if self.__last_idx < 0:
            raise Exception("No previous bounding box, nothing to compare")

        if bb.idx < 0:
            raise Exception(f"Illegal bounding box index {bb.idx}")

        if bb.idx < self.__last_idx:
            raise Exception("Can only check for subsequent frames")

        if abs(bb.idx - self.__last_idx) > max_idx_offset:
            return False

        previous = self.bounding_boxes[self.__last_idx]
        return previous.is_overlapping(bb, factor)

    def overlap_with_last_bb(self, bb: BoundingBox, max_idx_offset: int = 5) -> float:
        """
        Method allowing to check if a bounding box belongs to this track based on overlapping area of the given bounding box and the latest bounding box in the track
        :param bb: to be checked
        :param max_idx_offset: max supported frame index offset between current bb and latest bb of track
        :return: True if bb belongs to track, else False
        """
        if self.__last_idx < 0:
            raise Exception("No previous bounding box, nothing to compare")

        if bb.idx < 0:
            raise Exception(f"Illegal bounding box index {bb.idx}")

        if bb.idx < self.__last_idx:
            raise Exception("Can only check for subsequent frames")

        if abs(bb.idx - self.__last_idx) > max_idx_offset:
            return 0

        previous = self.bounding_boxes[self.__last_idx]
        return previous.get_overlap(bb)

