"""Heading source of the legacy PoseExtractor (same keyword as TimedPoseExtractor)."""
import datetime

import pytest
from pyproj import CRS, Transformer

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_from_srt_parser import AirDataFromSrtParser
from bambi.airdata.air_data_parser import AirDataParser
from bambi.webgl.pose_extractor import (
    AIRDATA_HEADING_SOURCES, SRT_HEADING_SOURCES, PoseExtractor, PoseExtractorCallback)
from bambi.webgl.timed_pose_extractor import HEADING_SOURCES

TRANSFORMER = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32617))


def _extractor(parser, **kw):
    return PoseExtractor(calibrated_frame_accessor=None, parser=parser,
                         rel_transformer=TRANSFORMER, **kw)


def test_sources_split_the_video_ones():
    assert set(AIRDATA_HEADING_SOURCES) | set(SRT_HEADING_SOURCES) == set(HEADING_SOURCES)


@pytest.mark.parametrize("parser,expected", [
    (AirDataParser(), "airdata_compass"),      # what use_gimbal_heading=False read
    (AirDataFromSrtParser(), "srt_drone"),     # compass_heading = SRT drone yaw
])
def test_default_keeps_the_old_heading(parser, expected):
    assert _extractor(parser).heading_source == expected


@pytest.mark.parametrize("parser,source", [
    (AirDataParser(), "srt_gimbal"),
    (AirDataParser(), "srt_drone"),
    (AirDataFromSrtParser(), "airdata_compass"),
    (AirDataFromSrtParser(), "airdata_gimbal"),
    (AirDataParser(), "magnetometer"),
])
def test_sources_of_the_other_file_are_rejected(parser, source):
    with pytest.raises(ValueError, match="not supported"):
        _extractor(parser, heading_source=source)


def test_use_gimbal_heading_is_gone():
    with pytest.raises(TypeError):
        _extractor(AirDataParser(), use_gimbal_heading=True)


def _frame():
    f = AirDataFrame()
    f.id, f.latitude, f.longitude, f.altitude = 0, 9.12, -79.68, 109.0
    f.compass_heading, f.gimbal_heading, f.gimbal_pitch = 250.0, 254.8, -90.0
    f.datetime = datetime.datetime(2025, 12, 8, 19, 38, 16)
    return f


@pytest.mark.parametrize("source,yaw", [
    ("airdata_compass", 250.0), ("srt_drone", 250.0),
    ("airdata_gimbal", 254.8), ("srt_gimbal", 254.8),
])
def test_callback_reads_the_chosen_heading(source, yaw, tmp_path):
    images = []
    cb = PoseExtractorCallback(
        images=images, frame=_frame(), offset_frame_time=0, is_airdata=False,
        rel_transformer=TRANSFORMER, target_folder=str(tmp_path), fpms=33.3,
        fovy_callback=lambda: 40.0, parse_yield=iter([_frame()]), apply_correction=False,
        drone_name=None, camera_name=None, heading_source=source, no_images=True)
    assert cb(0, None) is True
    assert images[0]["rotation"][2] == pytest.approx(yaw)
