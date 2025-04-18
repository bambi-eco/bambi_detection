import csv
from datetime import datetime
from typing import List

from bambi.airdata.air_data_frame import AirDataFrame


class AirDataWriter:
    """
    Class allowing to create AirData.csv files from AirDataFrames
    """

    def write(self, target_file: str, frames: List[AirDataFrame]) -> None:
        """
        Method for writing frames to a target file
        :param target_file: target file
        :param frames: to be written
        :return: None
        """
        columns = [
            "time(millisecond)",
            "datetime(utc)",
            "latitude",
            "longitude",
            "height_above_takeoff(feet)",
            "height_above_ground_at_drone_location(feet)",
            "ground_elevation_at_drone_location(feet)",
            "altitude_above_seaLevel(feet)",
            "height_sonar(feet)",
            "speed(mph)",
            "distance(feet)",
            "mileage(feet)",
            "satellites",
            "gpslevel",
            "voltage(v)",
            "max_altitude(feet)",
            "max_ascent(feet)",
            "max_speed(mph)",
            "max_distance(feet)",
            "xSpeed(mph)",
            "ySpeed(mph)",
            "zSpeed(mph)",
            "compass_heading(degrees)",
            "pitch(degrees)",
            "roll(degrees)",
            "isPhoto",
            "isVideo",
            "rc_elevator",
            "rc_aileron",
            "rc_throttle",
            "rc_rudder",
            "rc_elevator(percent)",
            "rc_aileron(percent)",
            "rc_throttle(percent)",
            "rc_rudder(percent)",
            "gimbal_heading(degrees)",
            "gimbal_pitch(degrees)",
            "gimbal_roll(degrees)",
            "battery_percent",
            "voltageCell1",
            "voltageCell2",
            "voltageCell3",
            "voltageCell4",
            "voltageCell5",
            "voltageCell6",
            "current(A)",
            "battery_temperature(f)",
            "altitude(feet)",
            "ascent(feet)",
            "flycStateRaw",
            "flycState",
            "message",
        ]
        frame_columns = []
        for column in columns:
            try:
                idx = column.index("(")
                frame_columns.append(column[:idx])
            except:
                frame_columns.append(column)

        with open(target_file, "w+", newline="", encoding="utf-8") as csvfile:
            spamwriter = csv.writer(
                csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            spamwriter.writerow(columns)
            for frame in frames:
                row = []
                for idx, column in enumerate(frame_columns):
                    full_column = columns[idx]
                    val = frame.__dict__.get(column)
                    if val is None:
                        row.append(None)
                    else:
                        if not isinstance(val, str):
                            if full_column.endswith("(feet)"):
                                val *= 3.28
                            elif full_column.endswith("(mph)"):
                                val /= 1.6093
                        if not isinstance(val, datetime):
                            row.append(f"{val}")
                        else:
                            row.append(val.strftime("%Y-%m-%d %H:%M:%S"))
                spamwriter.writerow(row)
