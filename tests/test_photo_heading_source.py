"""Heading source of the photo pose extractors (same keyword as TimedPoseExtractor)."""
import datetime

import pytest
from pyproj import CRS, Transformer

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.webgl.photo_pose_extractor import (
    PHOTO_HEADING_SOURCES, OrderedPhotoPoseExtractor, PhotoPoseExtractor,
    UniqueMatchPhotoPoseExtractor)

TRANSFORMER = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32617))


def _extractor(cls=PhotoPoseExtractor, **kw):
    return cls(rel_transformer=TRANSFORMER, fovy=40.0, **kw)


def _yaw(ex):
    frame = AirDataFrame()
    frame.latitude, frame.longitude, frame.altitude = 9.12, -79.68, 109.0
    frame.compass_heading, frame.gimbal_heading, frame.gimbal_pitch = 250.0, 254.8, -90.0
    frame.datetime = datetime.datetime(2025, 12, 8, 19, 38, 16)
    origin = AirDataFrame()
    origin.latitude, origin.longitude, origin.altitude = 9.12, -79.68, 49.0
    entry = ex._build_image_entry("a.jpg", frame, origin, TRANSFORMER.transform(9.12, -79.68))
    return entry["rotation"][2]


def test_supported_sources_are_the_flight_log_ones():
    assert PHOTO_HEADING_SOURCES == ("airdata_compass", "airdata_gimbal")


def test_default_is_the_compass():
    ex = _extractor()
    assert ex.heading_source == "airdata_compass"
    assert _yaw(ex) == pytest.approx(250.0)


def test_gimbal_heading():
    assert _yaw(_extractor(heading_source="airdata_gimbal")) == pytest.approx(254.8)


@pytest.mark.parametrize("source", ["srt_gimbal", "srt_drone", "magnetometer"])
def test_unsupported_sources_are_rejected(source):
    with pytest.raises(ValueError, match="not supported for photos"):
        _extractor(heading_source=source)


@pytest.mark.parametrize("cls", [OrderedPhotoPoseExtractor, UniqueMatchPhotoPoseExtractor])
def test_subclasses_take_the_same_keyword(cls):
    assert _extractor(cls, heading_source="airdata_gimbal").heading_source == "airdata_gimbal"


def test_use_gimbal_heading_is_gone():
    with pytest.raises(TypeError):
        _extractor(use_gimbal_heading=True)
