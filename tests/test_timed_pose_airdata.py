"""AirData slicing, heading source and heading check of TimedPoseExtractor."""
import datetime
import json

import numpy as np
import pytest

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.srt.srt_frame import SrtFrame
from bambi.webgl import timed_pose_extractor as tpe
from bambi.webgl.timed_pose_extractor import TimedPoseExtractor

T0 = datetime.datetime(2025, 12, 9, 1, 38, 0, tzinfo=tpe.TIMEZONE_UTC)


def _frames(n=60, video=range(20, 40), dt_ms=200, heading=250.0):
    out = []
    for i in range(n):
        f = AirDataFrame()
        f.time = i * dt_ms
        f.datetime = T0 + datetime.timedelta(milliseconds=f.time)
        f.isVideo = 1 if i in video else 0
        f.isPhoto = 0
        f.latitude = 9.12 + i * 1e-6
        f.longitude = -79.68
        f.altitude = 109.0
        f.height_above_takeoff = 60.0
        f.compass_heading = heading
        f.gimbal_heading = heading
        f.gimbal_pitch = -90.0
        out.append(f)
    return out


def _srt(n=10, start_ms=4000, dt_ms=200, gb_yaw=-102.8, drone_yaw=None):
    out = []
    for i in range(n):
        s = SrtFrame()
        s.id = i + 1
        s.timestamp = T0 + datetime.timedelta(milliseconds=start_ms + i * dt_ms)
        s.latitude, s.longitude, s.rel_alt = 9.12, -79.68, 60.0
        s.gb_yaw, s.gb_pitch = gb_yaw, -90.0
        s.drone_yaw = drone_yaw
        out.append(s)
    return out


def _extractor(**kw):
    return TimedPoseExtractor(None, **kw)


# --- flight log margin around the isVideo rows --------------------------------

def _slice(monkeypatch, frames, **kw):
    monkeypatch.setattr(tpe.AirDataParser, "parse", lambda self, path: frames)
    monkeypatch.setattr(tpe.AirDataParser, "get_frames_video_offset",
                        staticmethod(lambda frames, ts: 0))
    _, sliced = TimedPoseExtractor.get_air_data_frames(
        "unused.csv", T0, timezone=tpe.TIMEZONE_UTC, **kw)
    return [f.time for f in sliced]


def test_default_keeps_exactly_the_flagged_range(monkeypatch):
    # isVideo rows are 4000..7800 ms; default must stay the historic behaviour.
    times = _slice(monkeypatch, _frames())
    assert times[0] == 4000 and times[-1] == 7800


def test_slice_is_padded_beyond_the_isvideo_flag(monkeypatch):
    times = _slice(monkeypatch, _frames(), padding_ms=1000)
    assert times[0] == 3000 and times[-1] == 8800


def test_padding_stops_at_the_log_edges(monkeypatch):
    times = _slice(monkeypatch, _frames(video=range(2, 58)), padding_ms=3000)
    assert times[0] == 0 and times[-1] == 59 * 200


def test_negative_padding_is_rejected():
    with pytest.raises(ValueError):
        _extractor(airdata_video_padding_ms=-1)


# --- heading source ----------------------------------------------------------

def test_default_heading_source_is_the_compass():
    assert _extractor().heading_source == "airdata_compass"


def test_use_gimbal_heading_is_gone():
    # replaced by heading_source="airdata_gimbal"
    with pytest.raises(TypeError):
        _extractor(use_gimbal_heading=True)


def test_unknown_heading_source_is_rejected():
    with pytest.raises(ValueError):
        _extractor(heading_source="magnetometer")


def test_heading_per_source():
    ad = AirDataFrame()
    ad.compass_heading, ad.gimbal_heading = 250.0, 254.8
    srt = _srt(1, drone_yaw=-110.0)[0]
    assert _extractor(heading_source="airdata_compass")._heading(ad, srt) == 250.0
    assert _extractor(heading_source="airdata_gimbal")._heading(ad, srt) == 254.8
    assert _extractor(heading_source="srt_gimbal")._heading(ad, srt) == pytest.approx(257.2)  # -102.8
    assert _extractor(heading_source="srt_drone")._heading(ad, srt) == pytest.approx(250.0)   # -110


def test_srt_heading_falls_back_to_airdata_for_a_frame_without_it():
    ad = AirDataFrame()
    ad.compass_heading, ad.gimbal_heading = 250.0, 254.8
    srt = _srt(1, gb_yaw=None)[0]
    assert _extractor(heading_source="srt_gimbal")._heading(ad, srt) == 254.8
    assert _extractor(heading_source="srt_drone")._heading(ad, srt) == 250.0


def test_srt_drone_without_drone_yaw_fails_early():
    with pytest.raises(ValueError, match="drone_yaw"):
        _extractor(heading_source="srt_drone")._check_srt_heading(_srt(drone_yaw=None))
    _extractor(heading_source="srt_gimbal")._check_srt_heading(_srt(drone_yaw=None))  # gb_yaw present


# --- heading check -------------------------------------------------------------

def test_heading_check_flags_body_yaw_with_steady_gimbal():
    frames = _frames()
    for i, f in enumerate(frames):
        f.compass_heading = (358.0 + (5.0 if i % 2 else -5.0)) % 360  # wobble across north
    check = TimedPoseExtractor.heading_check(frames)
    assert check["prefer_gimbal_heading"]
    assert check["recommended_heading_source"] == "airdata_gimbal"
    assert 4.5 < check["compass_heading_std_deg"] < 5.5  # unwrapped, not ~180


def test_heading_check_quiet_when_gimbal_follows_body():
    frames = _frames()
    for i, f in enumerate(frames):
        f.compass_heading = f.gimbal_heading = 250.0 + i * 0.5  # a slow turn
    check = TimedPoseExtractor.heading_check(frames)
    assert not check["prefer_gimbal_heading"]
    assert check["recommended_heading_source"] is None


# --- headings in flight-log gaps filled from the SRT ---------------------------

def test_fill_missing_headings_interpolates_across_north():
    frames = _frames(n=5)
    frames[0].compass_heading, frames[4].compass_heading = 358.0, 6.0
    for f in frames[1:4]:
        f.compass_heading = None
    assert TimedPoseExtractor._fill_missing_headings(frames) == 3
    assert [round(f.compass_heading, 3) for f in frames] == [358.0, 0.0, 2.0, 4.0, 6.0]


def test_gap_frames_from_srt_without_drone_yaw_keep_the_heading(monkeypatch):
    ad = [f for f in _frames(n=40, video=range(40)) if not 10 <= f.time // 200 < 25]  # 3 s gap
    srt = _srt(n=40, start_ms=0, drone_yaw=None)
    monkeypatch.setattr(TimedPoseExtractor, "_get_srt_air_data_offset", staticmethod(lambda s, a: 0.0))
    _, new_ad, ranges = TimedPoseExtractor.adapt_srt_air_data_frames(srt, ad, ad[0])
    assert ranges, "the gap should have been filled from the SRT"
    converted = [f for f in new_ad if 1800 < f.time < 5000]
    assert converted
    assert all(f.compass_heading == pytest.approx(250.0) for f in converted)  # was None -> 0 (north)
    assert all(f.gimbal_heading == pytest.approx(257.2) for f in converted)   # -102.8 normalised


# --- end to end: poses.json ---------------------------------------------------------

class _FakeAccessor:
    def __init__(self, n):
        self.n = n

    def access(self, video_path, callback, check_duplicate_function=None):
        img = np.zeros((4, 4, 3), np.uint8)
        for i in range(self.n):
            if not callback(i, img):
                break


def _run(tmp_path, monkeypatch, **kw):
    frames = _frames()
    frames[19].isVideo = 0  # recording flag starts at row 20 (4000 ms)
    srt = _srt(n=12, start_ms=3600)  # first two video frames before the flag
    monkeypatch.setattr(TimedPoseExtractor, "get_srt_frames",
                        staticmethod(lambda paths, timezone=None: (srt, 1, [0] * len(srt))))
    monkeypatch.setattr(tpe.AirDataParser, "parse", lambda self, path: frames)
    monkeypatch.setattr(tpe.AirDataParser, "get_frames_video_offset", staticmethod(lambda f, ts: 0))
    monkeypatch.setattr(TimedPoseExtractor, "_get_srt_air_data_offset", staticmethod(lambda s, a: 0.0))
    ex = TimedPoseExtractor(_FakeAccessor(len(srt)), **kw)
    ex.extract(str(tmp_path), "ad.csv", "v.mp4", "v.srt", no_images=True, mask_images=False,
               remove_duplicates=False, timezone=tpe.TIMEZONE_UTC)
    return json.loads((tmp_path / "poses.json").read_text())


def test_poses_record_default_settings(tmp_path, monkeypatch):
    poses = _run(tmp_path, monkeypatch)
    assert poses["heading_source"] == "airdata_compass"
    assert poses["airdata_video_padding_ms"] == 0
    assert "heading_check" in poses
    assert all(img["rotation"][2] == pytest.approx(250.0) for img in poses["images"])


def test_poses_use_srt_gimbal_and_padding(tmp_path, monkeypatch):
    poses = _run(tmp_path, monkeypatch, heading_source="srt_gimbal", airdata_video_padding_ms=3000)
    assert poses["heading_source"] == "srt_gimbal"
    assert poses["airdata_video_padding_ms"] == 3000
    assert all(img["rotation"][2] == pytest.approx(257.2) for img in poses["images"])
