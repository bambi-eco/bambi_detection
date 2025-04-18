# pylint: disable=R0201
from datetime import datetime, timedelta
from io import TextIOBase
from typing import Optional, TextIO, Union, Sequence, Final, Collection

import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_parser import AirDataParser
from bambi.srt.srt_frame import SrtFrame


def interpolate_property(interp_x: ArrayLike, data_x: ArrayLike, ad_frames: Union[Sequence[object], Sequence[AirDataFrame]],
                         prop_name: str, period: Optional[float] = None) -> Sequence:
    """Interpolate a property of a dictionary-like object (e.g. AirDataFrame)

    :param interp_x: The x-coordinates to interpolate values for.
    :param data_x: The x-coordinates of the data points.
    :param ad_frames: The data points (as dicts or AirDataFrames) to interpolate from.
    :param prop_name: The property of the __dict__ to interpolate.
    :param period: A period for the x-coordinates. This parameter allows the proper interpolation of angular x-coordinates (e.g. 360°)

    :return The interpolated values.
    """

    if len(ad_frames) < 1:
        return []

    if len(ad_frames) == 1:
        return [getattr(ad_frames[0], prop_name)] * len(interp_x)

    convert_func = None

    property_values = [getattr(frame, prop_name) for frame in ad_frames]
    property_values = [prop_value if prop_value is not None else 0 for prop_value in property_values]
    property_values = np.array(property_values)

    if isinstance(property_values[0], datetime): # if the property is a datetime convert it to seconds
        first_datetime = property_values[0]
        property_values = np.array([(p - first_datetime).total_seconds() for p in property_values], dtype=float)

        def convert_func(delta):
            delta = 0.0 if np.isnan(delta) else float(delta)
            return first_datetime + timedelta(seconds=delta)

    if period is not None:
        property_values = np.unwrap(property_values, period=period)

    # using scipy interpolation (supports extrapolation)
    x_values = np.asarray(data_x, dtype=float)
    y_values = property_values
    interp_x_values = np.asarray(interp_x, dtype=float)
    interpolator = interpolate.interp1d(x_values, y_values, kind="linear", fill_value="extrapolate")
    interp_props = interpolator(interp_x_values)

    if period is not None:
        interp_props = np.mod(interp_props, period)

    if convert_func is not None:
        convert_func = np.vectorize(convert_func)
        interp_props = convert_func(interp_props)

    return interp_props.tolist()


class AirDataTimeInterpolator:
    """
    Parser that allows to read an AirData file
    """

    _ANGLE_PROPERTIES: Final[Collection[str]] = {
        "compass_heading",
        "gimbal_pitch",
        "gimbal_yaw",
        "gimbal_roll",
        "gimbal_heading",
    }

    def __init__(self, frames: Sequence[AirDataFrame]) -> None:
        if not any(frames):
            raise ValueError("No frames were read or passed.")

        self.frames = list(frames)

        for frame in self.frames:
            if frame.datetime is None or not isinstance(frame.datetime, datetime):
                raise ValueError("All frames must have a datetime of type datetime.datetime")

        self.start = self.frames[0].datetime
        self.seconds = np.array([(ad.datetime - self.start).total_seconds() for ad in self.frames])

    def __call__(self, time: Union[datetime, Sequence[datetime]]) -> Sequence[AirDataFrame]:
        if isinstance(time, datetime):
            time = (time,)

        x_seconds = np.array([(t - self.start).total_seconds() for t in time])

        targets = [AirDataFrame() for _ in range(len(time))]
        first_frame = self.frames[0].__dict__
        for prop_name, prop_value in first_frame.items():
            period = None
            if prop_name in self._ANGLE_PROPERTIES:
                period = 360  # we deal with angles (in degrees), so we need to unwrap them

            if isinstance(prop_value, (int, float, datetime)):
                new_values = interpolate_property(x_seconds, self.seconds, self.frames, prop_name, period=period)
            else:
                # for strings and other props take the first frame's value
                new_values = [first_frame[prop_name]] * len(time)

            # for isPhoto and isVideo, convert to binary (0 or 1)
            if prop_name in ("isPhoto", "isVideo"):
                new_values = np.array(new_values)
                new_values[new_values > 0] = 1

            for iv, target in enumerate(targets):
                setattr(target, prop_name, new_values[iv])

        return targets
