from typing import List

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.util.boundingbox_filter import BoundingBoxFilter


class AreaBoundingBoxFilter(BoundingBoxFilter):
    """
    Area based bounding box filter, which optionally also applies clipping for bounding boxes
    """
    def __init__(self, start_x: float, start_y: float, end_x: float, end_y: float, iou: float = 0.8, clip_boxes: bool = True):
        """
        Bounding box filter based on the definition of an area of interest
        :param start_x: x-coordinate of start
        :param start_y: y-coordinate of start
        :param end_x: x-coordinate of end
        :param end_y: y-coordinate of end
        :param iou: Expected IoU value for filtering
        :param clip_boxes: Flag if bounding boxes out of the area of interest should be clipped
        """
        self._bb = BoundingBox(-1, start_x, start_y, end_x, end_y)
        self._iou = iou
        self._clip_boxes = clip_boxes

    def filter_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        res = []
        for bb in boxes:
            if bb.is_overlapping(self._bb, self._iou):
                if self._clip_boxes:
                    start_x = max(self._bb.start_x, bb.start_x)
                    start_y = max(self._bb.start_y, bb.start_y)
                    end_x = min(self._bb.end_x, bb.end_x)
                    end_y = min(self._bb.end_y, bb.end_y)
                else:
                    start_x = bb.start_x
                    start_y = bb.start_y
                    end_x = bb.end_x
                    end_y = bb.end_y
                new_bb = BoundingBox(bb.idx, start_x, start_y, end_x, end_y, bb.label, bb.propability, bb.is_probagated, bb.visibility, bb.created_by)
                res.append(new_bb)
        return res


