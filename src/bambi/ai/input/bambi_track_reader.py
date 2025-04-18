import json
from typing import Union, List, Generator, Optional, Dict, Tuple

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.input.track_reader import TrackReader


class BambiTrackReader(TrackReader):
    def __init__(self, label_mapping: Optional[Dict[str, str]] = None):
        if label_mapping is None:
            self._label_mapping = {
                None: "Animal" ,
                "Q1219579": "Deer" ,
                "Q122069": "Roe Deer" ,
                "Q131340": "Chamois",
                "Q58697": "Wild Boar",
                "Q726": "Horse",
                "Q190516": "Sika Deer",
                "Q40435" : "Buffalo",
                "Q7368": "Sheep",
                "Q20908334": "Fallow Deer",
            }
        else:
            self._label_mapping = label_mapping
    def read_tracks(self, input_paths: Union[str, List[str]]) -> Generator[Track, None, None]:
        return (t for (f, t) in self.read_bambi_tracks(input_paths))

    def read_bambi_tracks(self, input_paths: Union[str, List[str]]) ->  Generator[Tuple[str, Track], None, None]:
        """
        Help method allowing to read bambi tracks
        :param input_paths: to be read
        :return: Generator for a tuple consisting of (Flight_name, Track)
        """
        paths = []
        if isinstance(input_paths, str):
            paths.append(input_paths)
        else:
            paths = input_paths

        for path in paths:
            with open(path) as file:
                curr = json.load(file)
                for track in curr:
                    label = self._label_mapping[track["animal"]]
                    confidence = track["confidence"]
                    res = Track()
                    for label_state in track["states"]:
                        idx = label_state["frameIdx"]
                        xs = []
                        ys = []
                        for c in label_state["pxlCoordinates"]:
                            xs.append(c["x"])
                            ys.append(c["y"])
                        start_x = min(xs)
                        start_y = min(ys)
                        end_x = max(xs)
                        end_y = max(ys)
                        is_propageted = label_state["isPropagated"]
                        bb = BoundingBox(idx, start_x, start_y, end_x, end_y, label, confidence, is_propageted, label_state["createdBy"])
                        res.add_bounding_box(bb)
                    yield track["flight"], res

