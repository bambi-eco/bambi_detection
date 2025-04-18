import abc
from typing import List, Generator, Union

from bambi.ai.domain.Track import Track


class TrackReader(abc.ABC):
    """
    Abstract class for reading tracks from a file representation
    """

    @abc.abstractmethod
    def read_tracks(self, input_paths: Union[str, List[str]]) -> Generator[Track, None, None]:
        pass