from enum import Enum
from typing import List

from bambi.domain.camera import Camera
from bambi.domain.manufacturer import Manufacturer


class Drone(Enum):
    M2EA = 0
    M3T = 1
    M30T = 2
    M300 = 3

    @classmethod
    def from_string(cls, value: str) -> "Drone":
        """
        Method for converting a string to an enum
        :param value: to be converted
        :return: Drone Enum Entry
        """
        value = value.lower()

        for man in Manufacturer:
            value = value.replace(str(man).lower(), "")

        value = value.replace("_", "")

        if value == "m2ea":
            return Drone.M2EA
        elif value in ["m3t", "m3te"]:
            return Drone.M3T
        elif value in ["m30t", "m30"]:
            return Drone.M30T
        elif value == "m300":
            return Drone.M300
        else:
            raise Exception(f"Unsupported drone type: {value}")
        
    def to_4str(self) -> str:
        if self==Drone.M3T:
            return "M3TE"
        else:
            return str(self)

    def __str__(self):
        if self == Drone.M2EA:
            return "M2EA"
        elif self == Drone.M3T:
            return "M3T"
        elif self == Drone.M30T:
            return "M30T"
        elif self == Drone.M300:
            return "M300"

    @classmethod
    def supported_cameras(cls, drone: "Drone") -> List[Camera]:
        """
        Returns the list of supported cameras of this drone
        :param drone: for which cameras should be received
        :return: List of Cameras
        """
        result = [Camera.Wide, Camera.Thermal]

        if drone in [Drone.M3T, Drone.M300, Drone.M30T]:
            result.append(Camera.Zoom)

        return result

    @classmethod
    def is_camera_supported(cls, drone: "Drone", camera: Camera) -> bool:
        """
        Method for checking if given camera is supported by drone
        :param drone: to be checked
        :param camera: to be checked
        :return: True IFF drone supports given camera
        """
        return camera in Drone.supported_cameras(drone)

    @classmethod
    def get_manufacturer(cls, drone: "Drone") -> Manufacturer:
        """
        Help method for getting the manufacturer of the given drone
        :param drone: for which manufacturer should be retrieved
        :return: Manufacturer of given drone
        """
        if drone in [Drone.M3T, Drone.M30T, Drone.M300, Drone.M2EA]:
            return Manufacturer.DJI

        raise Exception(f"Unknown manufacturer for drone {drone}")

    @classmethod
    def product_name(cls, drone: "Drone") -> str:
        """
        Help method for getting the product name (combination of manufacturer and drone name)
        :param drone: for which product name should be retrieved
        :return: Product name of given drone
        """
        return f"{Drone.get_manufacturer(drone)}_{drone}"
