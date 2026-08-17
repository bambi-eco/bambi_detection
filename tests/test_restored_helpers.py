# -*- coding: utf-8 -*-
"""The helpers restored in Phase 0 - pinned so they cannot go missing again."""
import datetime

import pytest

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.srt.srt_frame import SrtFrame
from bambi.util.paths import deviating_folders
from bambi.util.srt_air_data_converter import SrtAirDataConverter, _int_or_zero


# ---------------------------------------------------------------- paths

def test_deviating_folders_for_a_nested_file(tmp_path):
    sub = tmp_path / "flight" / "day2"
    sub.mkdir(parents=True)
    f = sub / "video.MP4"
    f.write_bytes(b"")
    rel = deviating_folders(str(tmp_path), str(f))
    assert rel.replace("\\", "/") == "flight/day2"


def test_deviating_folders_same_folder_is_empty(tmp_path):
    f = tmp_path / "video.MP4"
    f.write_bytes(b"")
    assert deviating_folders(str(tmp_path), str(f)) == ""


def test_deviating_folders_accepts_a_directory(tmp_path):
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert deviating_folders(str(tmp_path), str(sub)).replace("\\", "/") == "a/b"


# ---------------------------------------------------------------- srt -> airdata

def _srt():
    s = SrtFrame()
    s.id = 7
    s.timestamp = datetime.datetime(2025, 4, 5, 23, 59, 33, 895000)
    s.DiffTime = "33ms"
    s.latitude, s.longitude = -0.004274, 36.887871
    s.rel_alt, s.abs_alt = 16.565, 1817.096
    s.drone_speedx, s.drone_speedy, s.drone_speedz = 0.1, -0.2, 0.0
    s.drone_yaw, s.drone_pitch, s.drone_roll = 51.6, 1.5, -0.5
    s.gb_yaw, s.gb_pitch, s.gb_roll = 51.6, -90.0, 0.0
    return s


def test_convert_srt_maps_position_and_altitudes():
    ad = SrtAirDataConverter.convert_srt(_srt())
    assert isinstance(ad, AirDataFrame)
    assert (ad.latitude, ad.longitude) == (-0.004274, 36.887871)
    assert ad.height_above_takeoff == pytest.approx(16.565)
    assert ad.altitude == pytest.approx(16.565)
    assert ad.altitude_above_seaLevel == pytest.approx(1817.096)


def test_convert_srt_maps_gimbal_and_body_angles():
    ad = SrtAirDataConverter.convert_srt(_srt())
    assert (ad.gimbal_heading, ad.gimbal_pitch, ad.gimbal_roll) == (51.6, -90.0, 0.0)
    assert (ad.compass_heading, ad.pitch, ad.roll) == (51.6, 1.5, -0.5)


def test_convert_srt_keeps_identity_and_time():
    ad = SrtAirDataConverter.convert_srt(_srt())
    assert ad.id == 7
    assert ad.datetime == datetime.datetime(2025, 4, 5, 23, 59, 33, 895000)
    assert ad.time == 33
    assert ad.isVideo == 1


def test_convert_srt_tolerates_missing_fields():
    ad = SrtAirDataConverter.convert_srt(SrtFrame())
    assert ad.latitude is None and ad.gimbal_pitch is None
    assert ad.id == 0 and ad.time == 0


@pytest.mark.parametrize("raw,expected", [
    ("33ms", 33), (33, 33), (33.9, 33), (None, 0), ("", 0), ("ms", 0),
])
def test_difftime_normalisation(raw, expected):
    assert _int_or_zero(raw) == expected


def test_airdata_from_srt_parser_imports_again():
    """The whole point of restoring the converter."""
    from bambi.airdata.air_data_from_srt_parser import AirDataFromSrtParser
    assert AirDataFromSrtParser is not None
