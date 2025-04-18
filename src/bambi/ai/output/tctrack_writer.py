import json
from pathlib import Path
from typing import List, Dict, Tuple

from bambi.ai.domain.Track import Track
from bambi.ai.output.track_writer import TrackWriter


class TcTrackWriter(TrackWriter):
    """
    TrackWriter implementation for creating TcTrack (https://github.com/vision4robotics/TCTrack) datasets
    """

    def track_to_dict(self, detection: Track) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Help method allowing to convert a single track to a TcTrack dictionary
        :param detection: to be parsed
        :return:
        """
        res = {}
        for key in sorted(detection.bounding_boxes.keys()):
            bb = detection.bounding_boxes[key]
            if key > 999999:
                raise Exception("TcTrackWriter only intended for up to 999.999 frames")
            key = f"{key}"
            for i in range(len(key), 6):
                key = "0" + key

            res[key] = (int(bb.start_x), int(bb.start_y), int(bb.end_x), int(bb.end_y))
        return res

    def tracks_to_dict(self, input_path: str, detections: List[Track]) -> Dict[str, Dict[str, Dict[str, Tuple[int, int, int, int]]]]:
        """
        Help method allowing to convert a list of tracks to a TcTrack representation
        :param input_path: key in the result dictionary
        :param detections: to be parsed
        :return:
        """
        res = {}

        file_name = Path(input_path).stem
        for idx, detection in enumerate(detections):
            idx = f"{idx}"
            if len(idx) > 2:
                raise Exception("Only two digit idx allowed")
            if len(idx) < 2:
                idx = "0" + idx
            res[f"{idx}"] = self.track_to_dict(detection)

        return {f"{file_name}": res}

    def write_tracks(self, input_path: str, output_path: str, labels: List[str], detections: List[Track]):
        with open(output_path, "w") as f:
            json.dump(self.tracks_to_dict(input_path, detections), f)

    def write_dataset(self, output_path: str, detections_per_file: Dict[str, List[Track]]):
        """
        Help method allowing to write a TcTrack dataset to a labels file
        :param output_path: where to write the labels file
        :param detections_per_file: dataset that should be written
        :return:
        """
        res = {}
        for input_path, detections in detections_per_file.items():
            curr_res = self.tracks_to_dict(input_path, detections)
            for key in curr_res.keys():
                res[key] = curr_res[key]

        with open(output_path, "w") as f:
            json.dump(res, f)

    def write_track(self, input_path: str, output_path: str, labels: List[str], detection: Track):
        self.write_tracks(input_path, output_path, labels, [detection])