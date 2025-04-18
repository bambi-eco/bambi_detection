from typing import Optional

from bambi.domain.camera import Camera
from bambi.domain.drone import Drone
from bambi.domain.manufacturer import Manufacturer

default_Wide = (3840, 2160)
default_Thermal = (1280, 1024)


class SensorResolution:
    def __init__(
        self,
        drone: Drone = Drone.M30T,
        camera: Camera = Camera.Thermal,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):

        if width and height:
            self._width = width
            self._height = height

        else:
            if drone == Drone.M2EA:
                if camera == Camera.Wide:
                    self._width, self._height = default_Wide
                elif camera == Camera.Thermal:
                    self._width = 640
                    self._height = 512
                else:
                    raise Exception("Unsupported camera for this drone")

            elif drone == Drone.M3T:
                if camera == Camera.Wide:
                    self._width, self._height = default_Wide
                elif camera == Camera.Thermal:
                    self._width = 640
                    self._height = 512
                else:
                    raise Exception("Unsupported camera for this drone")

            elif drone == Drone.M30T:
                if camera == Camera.Wide:
                    self._width, self._height = default_Wide
                elif camera == Camera.Thermal:
                    self._width, self._height = default_Thermal
                else:
                    raise Exception("Unsupported camera for this drone")

            # elif drone == Drone.M300: # TODO: Add M300

            else:
                raise Exception("Unsupported drone")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self) -> tuple:
        return (self._width, self._height)
