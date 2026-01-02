import json
import os
import traceback
from typing import Optional, Callable, Generator, Tuple, Any, List

from ultralytics import YOLO

from bambi.ai.detection import Detection
from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.inference import model_registry
from bambi.ai.models.ultralytics_yolo import get_config
from bambi.ai.output.boundingbox_writer import BoundingBoxWriter
from bambi.ai.output.yolo_writer import YoloWriter
from bambi.video.video_frame_accessor import VideoFrameAccessor
import numpy.typing as npt


class UltralyticsYoloDetector(Detection):
    def __init__(self, model_path: str, labels: list[str], detection_writer: BoundingBoxWriter = YoloWriter(),
                 min_confidence: float = 0.7,
                 frame_accessor: Optional[Callable[[str], Generator[Tuple[int, npt.NDArray[Any]], None, None]]] = None,
                 verbose: bool = False):
        """
        :param model_name: the name of the model as configured in the BAMBI model config file
        :param min_confidence: Minimum confidence so a bounding box is valid
        :param frame_accessor: Callable for accessing the frames of a video based on the video's path
        """
        if frame_accessor is None:
            accessor = VideoFrameAccessor()
            self.__frame_accesor = accessor.access_yield
        else:
            self.__frame_accesor = frame_accessor

        self._min_confidence = min_confidence
        self._labels = labels
        self._model = YOLO(model_path, verbose=verbose)

        super().__init__(detection_writer, self._labels)

    def get_labels(self) -> List[str]:
        return self._labels.copy()

    def detect(self, input_path: str) -> Generator[Tuple[int, npt.NDArray[Any], List[BoundingBox]], None, None]:
        """
        Method for applying the model to the given input, creating the given output
        :param input_path: Input that should be analysed with AI inference
        :param output_path: Output created by AI model
        :return:
        """
        if input_path.lower().endswith(".mp4"):
            gen = self.__frame_accesor(input_path)
        else:
            gen = ((0, p) for p in [input_path])

        for idx, frame in gen:
            frame_boxes = self.detect_frame(idx, frame)
            if len(frame_boxes) > 0:
                yield idx, frame, frame_boxes

    def detect_frame(self, idx: int, frame: npt.NDArray[Any]) -> List[BoundingBox]:
        predictions = self._model.predict(frame)
        frame_boxes = []
        for prediction in predictions:
            if len(prediction.boxes) > 0:
                boxes = prediction.boxes
                for box in boxes:
                    prop = float(box.conf.item())
                    if prop >= self._min_confidence:
                        cls = prediction.names[int(box.cls.item())]
                        xyxy = box.xyxy.numpy()
                        bb = BoundingBox(idx, float(xyxy[0, 0]), float(xyxy[0, 1]), float(xyxy[0, 2]),
                                         float(xyxy[0, 3]), cls, prop, False, None, "")# self.__model_name)
                        frame_boxes.append(bb)
        return frame_boxes
