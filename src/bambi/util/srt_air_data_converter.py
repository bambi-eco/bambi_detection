# -*- coding: utf-8 -*-
"""Convert an SRT subtitle frame into an AirData-shaped frame.

DJI writes a subset of the flight log into the video's SRT track - position,
altitudes, gimbal angles, per-frame timestamp. For a flight without an AirData
export that is enough to derive poses, so the SRT frame is mapped onto the
``AirDataFrame`` fields the pose extractors already consume.

The original converter lived in the detached Pipeline project and never made it
into this repository, leaving ``AirDataFromSrtParser`` unimportable since the
2024 split; this is a fresh implementation of the same field mapping.
"""
from bambi.airdata.air_data_frame import AirDataFrame
from bambi.srt.srt_frame import SrtFrame


class SrtAirDataConverter:
    """Field mapping from :class:`SrtFrame` to :class:`AirDataFrame`."""

    @staticmethod
    def convert_srt(frame: SrtFrame) -> AirDataFrame:
        """Return an :class:`AirDataFrame` carrying *frame*'s data.

        Fields the SRT does not provide (battery, RC sticks, satellite count,
        ...) stay at their ``AirDataFrame`` defaults.

        :param frame: parsed SRT frame
        """
        ad = AirDataFrame()
        ad.id = frame.id if frame.id is not None else 0
        ad.datetime = frame.timestamp
        # DiffTime is the per-frame interval in ms; keep AirData's ``time``
        # (elapsed ms) if a caller wants to re-derive it downstream.
        ad.time = _int_or_zero(frame.DiffTime)

        ad.latitude = frame.latitude
        ad.longitude = frame.longitude
        # SRT ``rel_alt`` is height above the take-off point; ``abs_alt`` is
        # what DJI calls absolute altitude (effectively ellipsoidal).
        ad.height_above_takeoff = frame.rel_alt
        ad.altitude = frame.rel_alt
        ad.altitude_above_seaLevel = frame.abs_alt

        ad.xSpeed = frame.drone_speedx
        ad.ySpeed = frame.drone_speedy
        ad.zSpeed = frame.drone_speedz

        ad.compass_heading = frame.drone_yaw
        ad.pitch = frame.drone_pitch
        ad.roll = frame.drone_roll

        ad.gimbal_heading = frame.gb_yaw
        ad.gimbal_pitch = frame.gb_pitch
        ad.gimbal_roll = frame.gb_roll

        ad.isVideo = 1
        return ad


def _int_or_zero(value) -> int:
    """SRT ``DiffTime`` arrives as ``'33ms'`` or ``33``; normalise to an int."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0
