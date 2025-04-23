import abc
import json
import os
from pathlib import Path

import ultralytics
import yaml
from yaml import SafeLoader
from ultralytics import YOLO as u_yolo

ultralytics.checks()
def get_config(model_name: str = "yolov8N-20231023", verbose: bool = False):
    """

    :param model_name: the name of the model as configured in the BAMBI model config file
    :return:
    """
    bambi_models = os.environ.get('BAMBI_MODELS')

    if bambi_models is None:
        raise Exception("Environment variable BAMBI_MODELS not set!")

    with open(bambi_models) as model_json:
        model_config = json.load(model_json)

    ultralytic_model_config = model_config.get("ultralytics-yolo")
    if ultralytic_model_config is None:
        raise Exception(f"No configuration found for ultralytics-yolo models")

    model = ultralytic_model_config.get(model_name)
    if model is None:
        raise Exception(f"No model found with name {model_name}")
    model_parent_folder = os.path.normpath(Path(os.environ.get('BAMBI_MODELS')).parent)

    weights = model.get("weights")
    if weights is None:
        raise Exception("No weight file defined")
    weights = os.path.normpath(weights)

    classes = model.get("classes")
    if classes is None:
        raise Exception("No classes file defined")
    classes = os.path.normpath(classes)


    if not os.path.isabs(weights):
        weights = os.path.join(model_parent_folder, weights)

    if not os.path.isfile(weights):
        raise Exception(f"Given weights file can't be found {weights}")

    if not os.path.isabs(classes):
        classes = os.path.join(model_parent_folder, classes)

    if not os.path.isfile(classes):
        raise Exception(f"Given weights file can't be found {classes}")
    with open(classes) as f:
        data = yaml.load(f, Loader=SafeLoader)
        labels = list(data["names"].values())


    return model.get("datatype"), labels, weights, classes, u_yolo(weights, verbose=verbose)

