from enum import Enum


class Manufacturer(Enum):
    DJI = 0
    Yuneec = 1

    def __str__(self):
        if self == Manufacturer.DJI:
            return "DJI"
        elif self == Manufacturer.Yuneec:
            return "Yuneec"
