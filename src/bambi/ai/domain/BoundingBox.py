import math
from typing import Optional


class BoundingBox:
    """
    Class representing a single bounding box
    """

    def __init__(self, frame_idx: int, start_x: float, start_y: float, end_x: float, end_y: float,
                 label: Optional[str] = None,
                 propability: Optional[float] = None,
                 is_probagated: bool = False,
                 visibility: Optional[float] = None,
                 created_by: Optional[str] = None) -> None:
        """
        :param frame_idx: Idx of the frame, to which this bounding box belongs to
        :param start_x: top-left x coordinate
        :param start_y: top-left y coordinate
        :param end_x: bottom-right x coordinate
        :param end_y: bottom-right y coordinate
        :param label: label of bounding box
        :param propability: propability of bounding box
        :param is_probagated: Flag if bounding box is interpolated/propagated
        :param visibility: Visibility ratio, a number between 0 and 1 that says how much of that object is visible. Can be due to occlusion and due to image border cropping
        :param created_by: User id who created track
        """
        self.idx = frame_idx
        if start_x > end_x:
            self.start_x = end_x
            self.end_x = start_x
        else:
            self.start_x = start_x
            self.end_x = end_x
        if start_y > end_y:
            self.start_y = end_y
            self.end_y = start_y
        else:
            self.start_y = start_y
            self.end_y = end_y
        self.label = label
        self.propability = propability
        self.is_probagated = is_probagated
        self.visibility = visibility
        self.created_by = created_by

    def get_center(self) -> (int, int):
        """
        :return: Center position of this bounding box
        """
        return (self.end_x - self.start_x) / 2.0, (self.end_y - self.start_y) / 2.0 

    def get_center_on_image(self) -> (int, int):
        """
        :return: Center position of this bounding box on the image
        """
        return self.start_x + (self.end_x - self.start_x) / 2.0, self.start_y + (self.end_y - self.start_y) / 2.0

    def get_width(self) -> float:
        """
        :return: Width of bounding box
        """
        return self.end_x - self.start_x

    def get_height(self) -> float:
        """
        :return: Height of bounding box
        """
        return self.end_y - self.start_y

    def get_overlap(self, other: "BoundingBox") -> float:
        """
        Method allowing to check if two bounding boxes are overlapping
        :param other: to be checked
        :return: Overlap proportion of the two bounding boxes
        """
        x1_intersection = max(self.start_x, other.start_x)
        y1_intersection = max(self.start_y, other.start_y)
        x2_intersection = min(self.end_x, other.end_x)
        y2_intersection = min(self.end_y, other.end_y)

        # Calculate the areas of boxes 1 and 2
        area_box1 = (self.end_x - self.start_x) * (self.end_y - self.start_y)
        area_box2 = (other.end_x - other.start_x) * (other.end_y - other.start_y)

        # Calculate the area of intersection rectangle
        intersection_width = max(0.0, x2_intersection - x1_intersection)
        intersection_height = max(0.0, y2_intersection - y1_intersection)
        intersection_area = intersection_width * intersection_height

        # Calculate the Union area
        union_area = area_box1 + area_box2 - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def is_overlapping(self, other: "BoundingBox", factor: float = 0.8) -> bool:
        """
        Method allowing to check if two bounding boxes are overlapping
        :param other: to be checked
        :param factor: proportion of the area that has to be overlapping
        :return: True if this and other bounding box are overlapping with a proportion >= factor
        """
        return self.get_overlap(other) > factor

    def distance(self, other: "BoundingBox") -> float:
        """
        Calculate center based distance between two bounding boxes
        :param other: bounding box
        :return: distance between self and other bounding box
        """
        x1, y1 = self.get_center()
        x2, y2 = other.get_center()
        return abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def to_db_array(self) -> list[float]:
        return [self.start_x, self.start_y, self.end_x, self.start_y, self.end_x, self.end_y, self.start_x, self.end_y, self.start_x, self.start_y]