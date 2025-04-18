import csv
from typing import Union, List, Generator

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.input.track_reader import TrackReader


class MotReader(TrackReader):
    """
    TrackReader implementation for the MOT challenge format as described in https://arxiv.org/pdf/2003.09003.pdf
    """
    def __init__(self, labels: List[str]):
        self.__labels = labels
    def read_tracks(self, input_paths: Union[str, List[str]]) -> Generator[Track, None, None]:
        paths = []
        if isinstance(input_paths, str):
            paths.append(input_paths)
        else:
            paths = input_paths

        for path in paths:
            tracks = {}
            additional_tracks = []
            with open(path) as file:
                spamreader = csv.reader(file, delimiter=',')
                for row in spamreader:
                    if not row[0].isnumeric():
                        continue
                    frame_idx = int(row[0])
                    track_id = int(row[1])
                    bb_left = float(row[2])
                    bb_top = float(row[3])
                    bb_width = float(row[4])
                    bb_height = float(row[5])
                    bb_right = bb_left + bb_width
                    bb_bottom = bb_top + bb_height
                    confidence = float(row[6])
                    if confidence < 0:
                        confidence = None
                    label_id = int(row[7])
                    label = self.__labels[label_id] if label_id >= 0 else None
                    visibility = float(row[8])
                    if visibility < 0:
                        visibility = None

                    bb = BoundingBox(frame_idx, bb_left, bb_top, bb_right, bb_bottom, label, confidence, False, visibility)
                    if track_id < 0:
                        t = Track()
                        t.add_bounding_box(bb)
                        additional_tracks.append(t)
                    else:
                        if tracks.get(track_id) is None:
                            tracks[track_id] = Track()
                        tracks[track_id].add_bounding_box(bb)
            for track in tracks.values():
                yield track
            for track in additional_tracks:
                yield track