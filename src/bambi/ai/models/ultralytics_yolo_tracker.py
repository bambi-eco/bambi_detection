from typing import Optional, Callable, Generator, Tuple, Any, List

from bambi.ai.domain.BoundingBox import BoundingBox
from bambi.ai.domain.Track import Track
from bambi.ai.models.ultralytics_yolo import get_config
from bambi.ai.output.labelbox_writer import LabelboxWriter
from bambi.ai.output.track_writer import TrackWriter
from bambi.ai.tracking import Tracking
import numpy.typing as npt

from bambi.util.resource_utils import get_resource
from bambi.video.video_frame_accessor import VideoFrameAccessor


class UltralyticsYoloTracker(Tracking):
    def __init__(self, detection_writer: TrackWriter = LabelboxWriter(), model_name: str = "yolov8N-20231023",
                 frame_accessor: Optional[Callable[[str], Generator[Tuple[int, npt.NDArray[Any]], None, None]]] = None,
                 min_confidence: float = 0.7, min_iou: float = 0.5, tracker_file: str = None):
        """
        :param model_name: the name of the model as configured in the BAMBI model config file
        :param frame_accessor: Callable for accessing the frames of a video based on the video's path
        :param tracker_file: File containing the tracker configuration
        """
        if frame_accessor is None:
            accessor = VideoFrameAccessor()
            self.__frame_accesor = accessor.access_yield
        else:
            self.__frame_accesor = frame_accessor

        self.__model_name = model_name

        datatype, labels, weights, classes, model = get_config(model_name)

        self.__datatype = datatype
        self._labels = labels
        self.__weights = weights
        self.__classes = classes
        self._model = model
        self._min_confidence = min_confidence
        self._min_iou = min_iou

        if tracker_file is not None:
            self.__tracker_file = tracker_file
        else:
            self.__tracker_file = get_resource("ultralytics_bytetrack.yaml")

        super().__init__(detection_writer, labels)

    def track(self, input_path: str) -> List[Track]:
        if input_path.lower().endswith(".mp4"):
            gen = self.__frame_accesor(input_path)
        else:
            gen = ((0, p) for p in [input_path])

        res = {}
        non_tracked_boxes = []
        for idx, frame in gen:
            prediction = self._model.track(frame, persist=True, tracker=self.__tracker_file, conf=self._min_confidence, iou=self._min_iou)
            if len(prediction) > 0:
                for pred in prediction:
                    for box_idx, box in enumerate(pred.boxes):
                        prop = float(box.conf.item())
                        cls = pred.names[int(box.cls.item())]
                        xyxy = box.xyxy.numpy()
                        bb = BoundingBox(idx, float(xyxy[0, 0]), float(xyxy[0, 1]), float(xyxy[0, 2]),
                                         float(xyxy[0, 3]), cls, prop, False, None, self.__model_name)
                        if box.is_track:
                            track_id = int(pred.boxes.id[box_idx].item())
                            if res.get(track_id) is None:
                                res[track_id] = Track()
                            res[track_id].add_bounding_box(bb)
                        else:
                            t = Track()
                            t.add_bounding_box(bb)
                            non_tracked_boxes.append(t)

        return list(res.values()) + non_tracked_boxes
