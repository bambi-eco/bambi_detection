from enum import Enum
from typing import List


class Camera(Enum):
    Wide = 0
    Zoom = 1
    Thermal = 2
    FPV = 3

    @classmethod
    def from_string(cls, value: str) -> "Camera":
        """
        Method for converting a string to an enum
        :param value: to be converted
        :return: Camera Enum Entry
        """
        value = value.lower()

        if value in ["z", "zoom"]:
            return Camera.Zoom
        elif value in ["t", "thermal", "ir"]:
            return Camera.Thermal
        elif value in ["w", "v", "wide"]:
            return Camera.Wide
        elif value in ["s", "fpv", "self"]:
            return Camera.FPV
        raise Exception(f"Unknown camera {value}")

    def __str__(self):
        if self == Camera.Zoom:
            return "Z"
        elif self == Camera.Thermal:
            return "T"
        elif self == Camera.Wide:
            return "W"
        elif self == Camera.FPV:
            return "S"

    def valid_title_strings(self) -> List[str]:
        """
        :return: Valid substrings in DJI video titles
        """
        if self == Camera.Zoom:
            return ["_Z_", "_Z"]
        elif self == Camera.Thermal:
            return ["_T_", "_T"]
        elif self == Camera.Wide:
            return ["_W_", "_W", "_V_", "_V"]
        elif self == Camera.FPV:
            return ["_S_", "_S"]
        else:
            raise Exception("Not implemented camera")

    @classmethod
    def fullname(cls, camera: "Camera") -> str:
        """
        Help method for getting the full name of the camera
        :param camera: for which name should be retrieved
        :return: fullname
        """
        if camera == Camera.Zoom:
            return "Zoom"
        elif camera == Camera.Thermal:
            return "Thermal"
        elif camera == Camera.Wide:
            return "Wide"
        elif camera == Camera.FPV:
            return "FPV"
