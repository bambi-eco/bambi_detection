import json
from typing import Union, List, Generator

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.input.track_reader import TrackReader


class LabelboxReader(TrackReader):
    def read_tracks(self, input_paths: Union[str, List[str]]) -> Generator[Track, None, None]:
        paths = []
        if isinstance(input_paths, str):
            paths.append(input_paths)
        else:
            paths = input_paths

        for path in paths:
            with open(path) as file:
                curr = json.load(file)
                for segment in curr["segments"]:
                    track = Track()
                    for keyframe in segment["keyframes"]:
                        idx = keyframe["frame"]
                        start_x = keyframe["bbox"]["left"]
                        start_y = keyframe["bbox"]["top"]
                        end_x = keyframe["bbox"]["width"] + start_x
                        end_y = keyframe["bbox"]["height"] + start_y
                        label = keyframe["classifications"][0]["answer"]
                        bb = BoundingBox(idx, start_x, start_y, end_x, end_y, label, 1.0)
                        track.add_bounding_box(bb)
                    yield track

