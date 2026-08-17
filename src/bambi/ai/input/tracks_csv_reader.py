# -*- coding: utf-8 -*-
"""Reader for the geo-referenced tracks CSV the tracking scripts exchange.

Format, one row per detection::

    frame_id,track_id,x1,y1,z1,x2,y2,z2,confidence,class_id

An optional header row is tolerated and skipped.

``read_tracks_csv`` was lost in the 2024 repository clean-up while
``tracks_to_geojson`` kept importing it; restored here beside the other input
readers.
"""
import csv
from typing import Dict, List, Optional, Tuple

from bambi.georeferenced_tracking import Detection


def _maybe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None


def read_tracks_csv(path: str) -> Dict[int, List[Tuple[int, Detection]]]:
    """Read a geo-referenced tracks CSV.

    :param path: path to the CSV
    :return: ``track_id -> [(frame_id, Detection), ...]``, each list sorted by
        frame
    """
    tracks: Dict[int, List[Tuple[int, Detection]]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            frame_id = _maybe_int(row[0].strip())
            if frame_id is None:            # header row
                continue
            track_id = int(row[1].strip())
            x1, y1, z1 = (float(v) for v in row[2:5])
            x2, y2, z2 = (float(v) for v in row[5:8])
            conf = float(row[8])
            cls = int(float(row[9]))       # tolerant of "0.0"
            det = Detection(source_id=track_id, frame=frame_id,
                            x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                            conf=conf, cls=cls)
            tracks.setdefault(track_id, []).append((frame_id, det))

    for tid in tracks:
        tracks[tid].sort(key=lambda t: t[0])
    return tracks
