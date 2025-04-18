import datetime
from typing import Generator, List, Union

from bambi.srt.srt_frame import SrtFrame


class SrtMerger:
    __midnight = datetime.datetime(1, 1, 1, 0, 0, 0)

    def merge(
        self, srt_inputs: List[Union[Generator[SrtFrame, None, None], List[SrtFrame]]]
    ) -> List[SrtFrame]:
        """
        Method allowing to merge multiple srt files
        :param srt_inputs: Srt generator e.g. (SrtParser#parse_yield)
        :return: Merged srt inputs
        """
        res = []
        for srt in self.merge_yield(srt_inputs):
            res.append(srt)
        return res

    def merge_yield(
        self, srt_inputs: List[Union[Generator[SrtFrame, None, None], List[SrtFrame]]]
    ) -> Generator[SrtFrame, None, None]:
        """
        Method allowing to merge multiple srt files
        :param srt_inputs: Srt generator e.g. (SrtParser#parse_yield)
        :return: Merged srt inputs
        """
        current_id = 0
        current_end = None

        for idx, srt_input in enumerate(srt_inputs):
            if idx == 0:
                for frame in srt_input:
                    current_id = frame.id
                    current_end = frame.end
                    yield frame
            else:
                end = None
                for frame in srt_input:
                    if current_end is not None:
                        frame.start = self.__sum_times(frame.start, current_end)
                        frame.end = self.__sum_times(frame.end, current_end)

                    current_id += 1
                    frame.id = current_id
                    end = frame.end
                    yield frame
                current_end = end

    def __sum_times(self, t1: datetime.time, t2: datetime.time) -> datetime.time:
        """
        Help method for summing up to time objects
        :param t1: to be summed
        :param t2: to be summed
        :return: summed time deltas
        """
        delta = self.__time_to_timedelta(t1) + self.__time_to_timedelta(t2)
        return (SrtMerger.__midnight + delta).time()

    def __time_to_timedelta(self, t1: datetime.time) -> datetime.timedelta:
        """
        Help method for converting a time object to a timedelta object
        :param t1: to be converted
        :return: converted
        """
        return (
            datetime.datetime.combine(SrtMerger.__midnight, t1) - SrtMerger.__midnight
        )
