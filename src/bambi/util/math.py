import math
from math import cos, pi, radians, sin
from typing import List, Tuple


def point_pos(x0: float, y0: float, d: float, theta: float) -> List[float]:
    """
    Method for calculating a point based on a given point position, a distance and an angle
    :param x0: x of start position
    :param y0: y of start position
    :param d: distance between start and desired end position
    :param theta: angle of the vector (degrees)
    :return:
    """
    theta_rad = pi / 2 - radians(theta)
    return [x0 + d * cos(theta_rad), y0 + d * sin(theta_rad)]


def truncate(f: float, n: int = 0) -> float:
    """
    Method for truncating a float value to n digits without rounding
    :param f: to be truncated
    :param n: number of digits
    :return: truncated float
    """
    return math.floor(f * 10**n) / 10**n


def get_dms_from_decimal(decimal: float) -> Tuple[float, float, float]:
    """
    Convert decimal value to DMS (degrees, minutes, seconds) tuple
    :param decimal: to be converted
    :return:
    """
    degrees = truncate(decimal, 0)
    minutes_whole = (decimal - degrees) * 60
    minutes = truncate(minutes_whole, 0)
    seconds = (minutes_whole - minutes) * 60
    return degrees, minutes, seconds


def get_decimal_from_dms(dms: Tuple[float, float, float], ref: str = "N") -> float:
    """
    Method for converting a DMS (degrees, minutes, seconds) tuple to a decimal value
    :param dms: to be converted
    :param ref: compass direction (N, E, S, W)
    :return: decimal representation
    """
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ["S", "W"]:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return degrees + minutes + seconds
