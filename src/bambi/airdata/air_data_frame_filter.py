from typing import Generator, List

from bambi.airdata.air_data_frame import AirDataFrame


def get_film_frame(frames: List[AirDataFrame]) -> Generator[AirDataFrame, None, None]:
    """
    Simple generator pattern that allows to iterate a list of AirDataframes that are associated with film data (isVideo == 1)
    """

    for frame in frames:
        if frame.isVideo == 1:
            yield frame
