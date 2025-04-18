import json
import os
from typing import Optional, Callable, Generator, Tuple, Any, List

from bambi.ai.detection import Detection
from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.inference import model_registry
from bambi.ai.models.ultralytics_yolo import get_config
from bambi.ai.output.boundingbox_writer import BoundingBoxWriter
from bambi.ai.output.yolo_writer import YoloWriter
from bambi.video.video_frame_accessor import VideoFrameAccessor
import numpy.typing as npt


class UltralyticsYoloDetector(Detection):
    def __init__(self, detection_writer: BoundingBoxWriter = YoloWriter(), model_name: str = "yolov8N-20231023",
                 min_confidence: float = 0.7,
                 frame_accessor: Optional[Callable[[str], Generator[Tuple[int, npt.NDArray[Any]], None, None]]] = None):
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

        self.__model_name = model_name
        self._min_confidence = min_confidence

        datatype, labels, weights, classes, model = get_config(model_name)

        self.__datatype = datatype
        self._labels = labels
        self.__weights = weights
        self.__classes = classes
        self._model = model

        super().__init__(detection_writer, labels)

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
                                         float(xyxy[0, 3]), cls, prop, False, None, self.__model_name)
                        frame_boxes.append(bb)
        return frame_boxes

# Load all models
models = os.environ.get('BAMBI_MODELS')
if models is None:
    raise Exception("Environment variable BAMBI_MODELS not set!")
with open(models) as model_json:
    global_model_config = json.load(model_json)
    global_ultralytic_model_config = global_model_config.get("ultralytics-yolo")
    if global_ultralytic_model_config is None:
        raise Exception(f"No configuration found for ultralytics-yolo models")
    for key in global_ultralytic_model_config.keys():
        model_registry[key] = UltralyticsYoloDetector(model_name=key)
