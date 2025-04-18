import abc
from typing import List

from bambi.ai.domain.Track import Track


class TrackSimplifier(abc.ABC):
    """
    Algorithm allowing to simplify a track
    """

    def simplify_tracks(self, tracks: List[Track]) -> List[Track]:
        """
        Method for simplifying multiple tracks
        :param tracks: to be simplified
        :return: simplified tracks
        """
        res = []
        for track in tracks:
            res.append(self.simplify_track(track))
        return res

    @abc.abstractmethod
    def simplify_track(self, track: Track) -> Track:
        """
        Method for simplifying a single track
        :param track: to be simplified
        :return: simplified track
        """
        pass