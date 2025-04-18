from io import TextIOBase
from typing import Callable, Generator, List, Optional, TextIO, Union

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_parser import AirDataParserInterface
from bambi.srt.srt_frame import SrtFrame
from bambi.srt.srt_parser import SrtParser
from bambi.util.srt_air_data_converter import SrtAirDataConverter


class AirDataFromSrtParser(AirDataParserInterface):
    """
    Class that allows to read frames from an SRT file as air data frames
    """

    def __init__(self) -> None:
        self.srt_parser = SrtParser()

    def parse(
        self,
        file: Union[str, TextIO, TextIOBase],
        skip: int = 0,
        limit: Optional[int] = None,
    ) -> List[AirDataFrame]:
        """
        Method that reads an SRT file and converts its frames into AirData frames
        :param file: path or file pointer of the SRT file
        :param skip: Skip the first n frames and don't call the callback
        :param limit: Break reading frames after m frames
        :return: list of airdata frames
        """
        res = []

        def callback(frame: SrtFrame) -> None:
            res.append(SrtAirDataConverter.convert_srt(frame))

        self.srt_parser.parse_with_callback(file, callback, skip, limit)
        return res

    def parse_with_callback(
        self,
        file: Union[str, TextIO, TextIOBase],
        callback: Callable[[AirDataFrame], None],
        skip: int = 0,
        limit: Optional[int] = None,
    ) -> None:
        """
         Method that reads an SRT file and converts its frames into AirData frames
        :param file: path or file pointer of the SRT file
        :param callback: Callback that is executed for every frame
        :param skip: Skip the first n frames and don't call the callback
        :param limit: Break reading frames after m frames
        :return:
        """

        def inner_callback(frame: SrtFrame) -> None:
            callback(SrtAirDataConverter.convert_srt(frame))

        self.srt_parser.parse_with_callback(file, inner_callback, skip, limit)

    def parse_yield(
        self,
        file: Union[str, TextIO, TextIOBase],
        skip: int = 0,
        limit: Optional[int] = None,
    ) -> Generator[AirDataFrame, None, None]:
        """
        Method used to parse an AirData file, which will call the callback
        :param file: path or file pointer of the airdata file
        :param skip: Skip the first n frames and don't call the callback
        :param limit: Break reading frames after m frames
        :return:
        """
        for frame in self.srt_parser.parse_yield(file, skip, limit):
            yield SrtAirDataConverter.convert_srt(frame)
