from typing import List

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.util.util import chunker


class TrackInterpolator:
    """
    Class allowing to interpolate bounding boxes of a track
    """
    def interpolate(self, track: Track) -> Track:
        """
        Method to interpolate track
        :param track: to be interpolated
        :return: Interpolated (new) track
        """
        res = Track()
        res.track_id = track.track_id

        first_idx = True
        last_idx = None
        for idx in track.bounding_boxes.keys():
            if first_idx:
                last_idx = idx
                first_idx = False
                continue

            last_bb = track.bounding_boxes.get(last_idx)
            res.add_bounding_box(last_bb)
            if idx == last_idx + 1:
                last_idx = idx
                continue

            curr_bb = track.bounding_boxes.get(idx)

            diff = idx - last_idx

            for intermediate_idx in range(1, diff):
                weight = intermediate_idx / diff
                start_x = last_bb.start_x + (curr_bb.start_x - last_bb.start_x) * weight
                start_y = last_bb.start_y + (curr_bb.start_y - last_bb.start_y) * weight
                end_x = last_bb.end_x + (curr_bb.end_x - last_bb.end_x) * weight
                end_y = last_bb.end_y + (curr_bb.end_y - last_bb.end_y) * weight

                if curr_bb.propability is None or last_bb.propability is None:
                    prob = None
                else:
                    prob = last_bb.propability + (curr_bb.propability - last_bb.propability) * weight

                if curr_bb.visibility is None or last_bb.visibility is None:
                    visibility = None
                else:
                    visibility = last_bb.visibility + (curr_bb.visibility - last_bb.visibility) * weight

                new_bb = BoundingBox(last_idx + intermediate_idx, start_x, start_y, end_x, end_y, last_bb.label, prob, True, visibility)
                res.add_bounding_box(new_bb)
            last_idx = idx

        last_bb = track.bounding_boxes.get(last_idx)
        if last_bb is not None:
            res.add_bounding_box(last_bb)

        return res

    def interpolate_tracks(self, tracks: List[Track]) -> List[Track]:
        """
        :param tracks: Tracks to be interpolated
        :return: interpolated tracks
        """
        return [self.interpolate(x) for x in tracks]