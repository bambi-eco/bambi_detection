from io import TextIOBase
from typing import Generator, Optional, TextIO, Union

from bambi.airdata.air_data_frame import AirDataFrame, interpolate_frames
from bambi.airdata.air_data_parser import AirDataParser


class InterpolatedAirDataParser(AirDataParser):
    """
    AirData Parser used to interpolate between the frames
    """

    def __init__(self, target_fps: float, delimiter: str = ","):
        """
        Constructor of an interpolated AirData Parser
        :param target_fps: the fps used as target value
        """
        super().__init__(delimiter)
        self.__target_fps = target_fps

    def parse_yield(
        self,
        file: Union[str, TextIO, TextIOBase],
        skip: int = 0,
        limit: Optional[int] = None,
    ) -> Generator[AirDataFrame, None, None]:
        generator = super().parse_yield(file)
        previous_frame = next(generator)
        frame_count = 1
        if skip <= 0:
            # return first frame
            yield previous_frame

        if previous_frame is not None:
            frame_difference = 1000 / self.__target_fps
            current_frame_ms = frame_difference
            for frame in generator:
                while current_frame_ms <= frame.time:
                    frame_count += 1
                    if frame_count > skip:
                        interpolation_distance = current_frame_ms - previous_frame.time
                        target_distance = frame.time - previous_frame.time
                        # if target_distance == 0 we found a duplicated frame and skip it
                        if target_distance != 0.0:
                            weight = interpolation_distance / target_distance
                            interpolated_frame = interpolate_frames(
                                previous_frame, frame, weight
                            )
                            # fix invalid GPS information if possible
                            if previous_frame.longitude == 0.0:
                                interpolated_frame.longitude = frame.longitude
                            elif frame.longitude == 0.0:
                                interpolated_frame.longitude = previous_frame.longitude
                            if previous_frame.latitude == 0.0:
                                interpolated_frame.latitude = frame.latitude
                            elif frame.latitude == 0.0:
                                interpolated_frame.latitude = previous_frame.latitude
                            interpolated_frame.time = current_frame_ms
                            yield interpolated_frame
                    if limit is not None and frame_count >= skip + limit:
                        return
                    current_frame_ms += frame_difference

                previous_frame = frame
        else:
            yield None
