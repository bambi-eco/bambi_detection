import json
from typing import List, Optional, Callable, Dict

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.output.track_writer import TrackWriter


class LabelboxWriter(TrackWriter):
    """
    Writer for labelbox's boundin box format (https://docs.labelbox.com/reference/import-video-annotations)
    """
    def __init__(self, label_function: Optional[Callable[[Track], str]] = None):
        if label_function is None:
            self._label_function = lambda x: x.get_class()
        else:
            self._label_function = label_function

    def _to_frame(self, bb: BoundingBox) -> Dict[str, any]:
        return {"top": bb.start_y, "left": bb.start_x, "height": bb.get_width(), "width": bb.get_height()}

    def write_tracks(self, input_path: str, output_path: str, labels: List[str], detections: List[Track]):
        if not output_path.endswith(".json"):
            output_path += ".json"

        with open(output_path, "w") as file:
            segments = []
            res = {"name": input_path, "segments": segments}
            for detection in detections:
                label = self._label_function(detection)
                frame = {}
                segments.append(frame)
                keyframes = []
                frame["keyframes"] = keyframes
                for idx in sorted(detection.bounding_boxes.keys()):
                    bb = detection.bounding_boxes.get(idx)
                    curr = {"frame": bb.idx,
                            "bbox": self._to_frame(bb),
                            "classifications": [{"name": "bbox_radio", "answer": label}]
                            }
                    keyframes.append(curr)
            json.dump(res, file)

    def write_track(self, input_path: str, output_path: str, labels: List[str], detection: Track):
        self.write_tracks(input_path, output_path, labels, [detection])