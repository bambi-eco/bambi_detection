# pylint: disable=R0201
import base64
import datetime
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional, Set

import requests

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.airdata.air_data_parser import AirDataParser


class DetailLevel(Enum):
    """
    The amount of information to include in the response:
        - basic for basic information
        - comprehensive for extended information
    """

    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"


class Roles(Enum):
    """
    Airdata Roles
    """

    PILOT = "Pilot-in-Command"
    MISSION_COMMANDER = "Mission+Commander"
    CONTROL_PILOT = "Pilot-at-Controls"
    SENSOR_OPERATOR = "Sensor+Operator"
    VISUAL_OBSERVER = "Visual+Observer"
    INSTRUCTOR = "Instructor"


class AirDataClient:
    """
    HTTP Client that allows to access the AirData API (https://app.airdata.com/docs/api/)
    Requires an enterprise account
    """

    def __init__(self, api_key: str, flight_upload_key: Optional[str] = None):
        """
        :param api_key: ApiKey of your AirData.com enterprise account
        :param flight_upload_key: ApiKey for AirData.com's Flight Upload API
        """

        self.api_key = api_key
        self.flight_upload_key = flight_upload_key
        self.__base_api_url = "https://api.airdata.com/flights"

    def get_authorization_header(self) -> str:
        """
        :return: Authorization header required for AirData API
        """
        message_bytes = f"{self.api_key}:".encode("UTF-8")
        return base64.b64encode(message_bytes).decode("UTF-8")

    def get_flights_to_air_data_csv(self, flights: Dict[str, Any]) -> List[Any]:
        """
        Method for getting the AirData CSV's for flights retrieved using self.get_flights()
        :param flights: Result of self.get_flights()
        :return: List of CSV files
        """
        parser = AirDataParser()
        data = flights.get("data")
        res: List[List[AirDataFrame]] = []
        if data is not None:
            for flight in data:
                link = flight.get("csvLink")
                r = requests.get(link)
                with StringIO(r.text) as f:
                    res.append(parser.parse(f))
        return res

    def get_drones(self):
        # Add basic http authorization header
        headers = {"Authorization": f"Basic {self.get_authorization_header()}"}

        # create your own params string, so requests does not do a URL encoding...
        r = requests.get("https://api.airdata.com/drones", headers=headers)
        return r.json()

    def get_drone_by_serial(self, serial_id: str) -> Optional[dict[str, any]]:
        drones = self.get_drones()
        target_drone = None
        for drone in drones:
            if drone.get("internalSerial") == serial_id:
                target_drone = drone
                break
        return target_drone


    def get_flights(
        self,
        radius: Optional[int] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        detail_level: Optional[DetailLevel] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        drone_ids: Optional[List[str]] = None,
        battery_ids: Optional[List[str]] = None,
        role_ids: Optional[Set[Roles]] = None,
        pilot_ids: Optional[Set[str]] = None,
        only_without_checklist: Optional[bool] = None,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        The API will return a list of flights, and it accepts optional search parameters, like flight time range, address, pilot, associated equipment (drone or battery), etc.

        :param detail_level: The amount of information to include in the response
        :param latitude: Search Latitude
        :param longitude: Search Longitude
        :param radius: An integer that represent the search area radius in miles or kilometers, depending on the user unit configuration. Used with either address or latitude & longitude.
        :param start: Formatted date - show flights that started after this date. Flight time is searched by local flight time. Example: start=2019-05-01+00:00:00
        :param end: show flights that started before this date. Example: end=2019-05-01+00:00:00
        :param limit: The argument accept integer value that represent the maximum limit of flights that should return. Default: 30 Maximum: 100
        :param offset: Retrieve flights from a certain offset. For example, for page 2, use: offset=100&limit=100
        :param drone_ids: List of drone IDs that you want to search on. Only flights done with these drones will be listed. See the Drone List API on how to find a specific ID.
        :param battery_ids: List of battery IDs that you want to search on. Only flights done with these batteries will be listed. See the Battery List API on how to find a specific ID.
        :param pilot_ids: List of pilot IDs (similar to drone_ids/battery_ids above).
        :param role_ids: Comma separated string of which roles to filter by, and works with the pilot_ids above.
        :param only_without_checklist: The parameter accept 0 or 1. Specify 1 to only get flights without an answered checklist.
        :param address: Address as a string value. For example: address=4364+Town+Center+Blvd+,El+Dorado+Hills+,CA
        :return: Result JSON of AirData api
        """
        if latitude is not None and longitude is not None and address is not None:
            raise Exception(
                "Address can't be combined with latitude/longitude. Either the one or the other is required"
            )

        if (latitude is not None and longitude is None) or (
            latitude is None and longitude is not None
        ):
            raise Exception(
                "If either latitude or longitude is given, also the other parameters must be given."
            )

        payload: Dict[str, Any] = {}

        if detail_level is not None:
            payload["detail_level"] = detail_level.value

        if limit is not None:
            payload["limit"] = limit

        if address is not None:
            payload["address"] = address

        if latitude is not None and longitude is not None:
            payload["latitude"] = str(latitude)
            payload["longitude"] = str(longitude)

        if radius is not None:
            payload["radius"] = str(radius)

        if start is not None:
            payload["start"] = start.strftime("%Y-%m-%d+%H:%M:%S")

        if end is not None:
            payload["end"] = end.strftime("%Y-%m-%d+%H:%M:%S")

        if drone_ids is not None:
            payload["drone_ids"] = ",".join(drone_ids)

        if battery_ids is not None:
            payload["battery_ids"] = ",".join(battery_ids)

        if pilot_ids is not None:
            payload["pilot_ids"] = ",".join(pilot_ids)

        if role_ids is not None:
            payload["role_ids"] = ",".join([x.value for x in role_ids])

        if only_without_checklist is not None:
            payload["only_without_checklist"] = 1 if only_without_checklist else 0

        if limit is not None:
            payload["limit"] = limit

        if offset is not None:
            payload["offset"] = offset

        # Add basic http authorization header
        headers = {"Authorization": f"Basic {self.get_authorization_header()}"}

        # create your own params string, so requests does not do a URL encoding...
        payload_str = "&".join("%s=%s" % (k, v) for k, v in payload.items())
        r = requests.get(self.__base_api_url, params=payload_str, headers=headers)
        return r.json()
