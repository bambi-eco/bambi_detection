from typing import List

from bambi.ai.domain.Track import Track
from bambi.ai.output.track_writer import TrackWriter


class MotWriter(TrackWriter):
    """
    TrackWriter implementation based on the MOT challenge format as described in https://arxiv.org/pdf/2003.09003.pdf
    """

    def __init__(self, include_header: bool = False):
        """
        :param include_header: Flag if header should be included when creating CSV label file
        """
        self.__include_header = include_header

    def write_tracks(self, input_path: str, output_path: str, labels: List[str], detections: List[Track]):
        label_mapping = {}
        for idx, label in enumerate(labels):
            label_mapping[label] = idx


        bounding_boxes = {}
        for idx, track in enumerate(detections):
            for frame_idx, bb in track.bounding_boxes.items():
                if bounding_boxes.get(frame_idx) is None:
                    bounding_boxes[frame_idx] = []
                bounding_boxes[frame_idx].append((idx, bb))

        with open(output_path, "w") as file:
            if self.__include_header:
                file.write("frame number, identity number, bounding box left, bounding box top, bounding box width, bounding box height, confidence score, class, visibility")

            for frame_idx in sorted(bounding_boxes.keys()):
                bbs_and_track_id = bounding_boxes.get(frame_idx)
                for track_id, bb in bbs_and_track_id:
                    prob = bb.propability if bb.propability is not None else -1
                    vis = bb.visibility if bb.visibility is not None else -1
                    label = label_mapping.get(bb.label)
                    if label is None:
                        label = -1
                    file.write(f"{frame_idx},{track_id},{bb.start_x},{bb.start_y},{bb.get_width()},{bb.get_height()},{prob},{label},{vis}\n")

    def write_track(self, input_path: str, output_path: str, labels: List[str], detection: Track):
        self.write_tracks(input_path, output_path, labels, [detection])