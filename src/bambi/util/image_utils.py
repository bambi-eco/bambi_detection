from typing import Any, Optional

import cv2
import numpy as np
import numpy.typing as npt


def equal_images(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> bool:
    """
    Method allowing to check if two images are equal
    :param frame1: reference
    :param frame2: to be compared
    :return: true iff images are equal in all channels
    """
    difference = cv2.subtract(frame1, frame2)
    channels = cv2.split(difference)
    for channel in channels:
        if cv2.countNonZero(channel) != 0:
            return False
    return True


def image_equality(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> float:
    """
    Method allowing to check how equal images are (in percentage)
    :param frame1: reference
    :param frame2: to be compared
    :return: 0 iff images are equal in all channels; and 1 if not one pixel is equal
    """
    difference = cv2.subtract(frame1, frame2)
    channels = cv2.split(difference)
    different_pixels = 0
    for channel in channels:
        different_pixels += cv2.countNonZero(channel)

    total_pixels = 1
    for shape in frame1.shape:
        total_pixels *= shape

    return different_pixels / total_pixels


def image_equality_check(
    frame1: npt.NDArray[Any], frame2: npt.NDArray[Any], equality_threshold: float = 0.05
) -> bool:
    """
    Method allowing to check how equal images are (in percentage)
    :param frame1: reference
    :param frame2: to be compared
    :param equality_threshold: Threshold used to determine if images are equal. Default: 5%
    :return: true iff equality_threshold is not exceeded
    """
    return image_equality(frame1, frame2) < equality_threshold


def ssd(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> int:
    """
    Method for calculating the sum of squared error of two images
    :param frame1: reference
    :param frame2: to be compared
    :return: Sum of squared error
    """
    if len(frame1.shape) == 3:
        channels = frame1.shape[2]
        return np.sum((frame1[:, :, 0:channels] - frame2[:, :, 0:channels]) ** 2)
    else:
        return np.sum((frame1[:, :] - frame2[:, :]) ** 2)


def ssd_equal_images(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> bool:
    """
    Method allowing to check if two images are equal based on the sum of squared error metric
    :param frame1: reference
    :param frame2: to be compared
    :return: true iff images are equal in all channels
    """
    return ssd(frame1, frame2) == 0


def image_resize(
    image: npt.NDArray[Any],
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_AREA,
) -> npt.NDArray[Any]:
    """
    Resizes an image to a given width and height. If only one of the two is specified, the image is resized proportionally.
    :param image (npt.NDArray[Any]): image to be resized
    :param width (Optional[int], optional): width of the resized image. Defaults to None.
    :param height (Optional[int], optional): height of the resized image. Defaults to None.
    :param inter (cv2.INTER_AREA, optional): interpolation method. Defaults to cv2.INTER_AREA.
    :return: resized image
    """
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then just return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None and height is not None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif width is not None and height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        # this should not happen
        raise ValueError("resize_image: something went wrong!")

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
