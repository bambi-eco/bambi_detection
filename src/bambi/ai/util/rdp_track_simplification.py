import copy

import numpy as np
import rdp

from bambi.ai.domain.Track import Track
from bambi.ai.util.track_simplification import TrackSimplifier


class RdpTrackSimplifier(TrackSimplifier):
    """
    Track simplification based on the Ramer-Douglas-Peucker Algorithm
    """
    def __init__(self, epsilon: float =0):
        """
        :param epsilon: epsilon in the rdp algorithm
        """
        self._epsilon = epsilon

    def simplify_track(self, track: Track) -> Track:
        centers = []
        keys = sorted(track.bounding_boxes.keys())
        for bb_idx in keys:
            bb = track.bounding_boxes.get(bb_idx)
            centers.append(np.asarray(bb.get_center()))

        masks = rdp.rdp_iter(centers, self._epsilon, return_mask=True)
        t = Track()
        for idx, mask in enumerate(masks):
            if mask:
                bb_idx = keys[idx]
                bb = copy.deepcopy(track.bounding_boxes.get(bb_idx))
                t.add_bounding_box(bb)

        return t
