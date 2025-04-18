import json
from typing import List, Dict, Optional

from bambi.ai.domain.Track import Track
from bambi.ai.output.track_writer import TrackWriter


class BambiTrackWriter(TrackWriter):
    """
    Track writer allowing to create a BAMBI Backend compatible json representation of tracks
    """
    def __init__(self, flight: str, label_mapping: Optional[Dict[str, str]] = None):
        """
        :param flight: name of the flight to be used in the labels file
        :param user_id: user id used in the labels file
        :param label_mapping: dictionary allowing to map model labels to backend animal ids (wikidata ids)
        """
        self._flight = flight
        if label_mapping is None:
            self._label_mapping = {
                "Animal": None,
                "Deer": "Q1219579",
                "Roe Deer": "Q122069",
                "Chamois": "Q131340",
                "Wild Boar": "Q58697",
                "Rabbit": "Q9394",
                "Horse": "Q726",
                "Sika Deer": "Q190516",
                "Buffalo": "Q40435",
                "Sheep": "Q7368",
                "Fallow Deer": "Q20908334",
            }
        else:
            self._label_mapping = label_mapping

    def write_tracks(self, input_path: str, output_path: str, labels: List[str], detections: List[Track]):
        """
        :param input_path: Source of data
        :param output_path: Target to write detections
        :param labels: To be expected
        :param detections: To be written
        :return: None
        """
        if not output_path.endswith(".json"):
            output_path += ".json"

        res = []

        for detection in detections:
            curr = {
                "flight": self._flight,
                "animal": self._label_mapping.get(detection.get_class()),
                "age": 0,
                "gender": 0,
                "confidence": detection.get_confidence()
            }
            res.append(curr)
            states = []
            curr["states"] = states
            for bb_idx in sorted(detection.bounding_boxes.keys()):
                bb = detection.bounding_boxes.get(bb_idx)
                start_x = int(bb.start_x)
                end_x = int(bb.end_x)
                start_y = int(bb.start_y)
                end_y = int(bb.end_y)
                pxlCoordinates = [
                    {"x": start_x,
                     "y": end_y},
                    {"x": end_x,
                     "y": end_y},
                    {"x": end_x,
                     "y": start_y},
                    {"x": start_x,
                     "y": start_y},
                    {"x": start_x,
                     "y": end_y}
                ]

                states.append({
                    "visibility": bb.visibility,
                    "frameIdx": bb_idx,
                    "isCurrentlyForwarded": False,
                    "coordinates": [],
                    "pxlCoordinates": pxlCoordinates,
                    "createdBy": bb.created_by,
                    "isPropagated": bb.is_probagated
                })

        with open(output_path, "w") as file:
            json.dump(res, file)


    def write_track(self, input_path: str, output_path: str, labels: List[str], detection: Track):
        """
        :param input_path: Source of data
        :param output_path: Target to write detections
        :param labels: To be expected
        :param detection: To be written
        :return: None
        """
        self.write_tracks(input_path, output_path, labels, [detection])
