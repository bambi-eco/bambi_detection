import abc
from typing import List

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track


class BoundingBoxFilter(abc.ABC):
    """
    Class allowing to filter bounding boxes
    """
    @abc.abstractmethod
    def filter_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Method allowing to filter bounding boxes
        :param boxes: to be filtered
        :return: Filtered list of bounding boxes
        """
        pass
    def filter_tracks(self, tracks: List[Track]) -> List[Track]:
        """
        Help method for filtering tracks
        :param tracks: tog be filtered
        :return: List of filtered tracks
        """
        res = []
        for track in tracks:
            bbs = self.filter_boxes(track.bounding_boxes.values())
            if len(bbs) > 0:
                t = Track()
                for bb in bbs:
                    t.add_bounding_box(bb)
                res.append(t)
        return res