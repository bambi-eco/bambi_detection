from typing import List, Any, Tuple

import cv2
import numpy as np

import numpy.typing as npt

from bambi.ai.visualization.line_style import LineStyle


def draw_line(img: npt.NDArray[Any], pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int],
              thickness: int = 1, style: LineStyle = LineStyle.DASHED, gap: int = 20):
    """
    Help method for drawing a line
    :param img: image to draw
    :param pt1: start point of line
    :param pt2: end point of line
    :param color: color to be drawn
    :param thickness: line thickness
    :param style: line drawing style
    :param gap: Gap between dots/dashes (only used if style is not SOLID)
    :return: None
    """
    if style is LineStyle.SOLID:
        cv2.line(img, pt1, pt2, color, thickness)
    else:
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        if style is LineStyle.DOTTED:
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1


def draw_poly(img: npt.NDArray[Any], pts: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int = 1,
              style: LineStyle = LineStyle.DASHED, gap: int = 20):
    """
    Help method for drawing a polygon
    :param img: image to draw
    :param pts: points of the polygon to be drawn
    :param color: color to be drawn
    :param thickness: line thickness
    :param style: line draw style
    :param gap: Gap between dots/dashes
    :return:
    """
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_line(img, s, e, color, thickness, style, gap)


def draw_rect(img: npt.NDArray[Any], pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int],
              thickness: int = 1, style: LineStyle = LineStyle.DASHED, gap: int = 20):
    """
    Help method for drawing a rectangle
    :param img: image to draw
    :param pt1: start point of line
    :param pt2: end point of line
    :param color: color to be drawn
    :param thickness: line thickness
    :param style: line draw style
    :param gap: Gap between dots/dashes
    :return: None
    """
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_poly(img, pts, color, thickness, style, gap)