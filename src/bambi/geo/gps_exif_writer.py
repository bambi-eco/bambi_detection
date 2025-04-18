import datetime
from typing import Optional, Tuple

from PIL import Image
from PIL.ExifTags import GPSTAGS

from bambi.util.math import get_decimal_from_dms, get_dms_from_decimal


class GpsExifWriter:
    """
    Class allowing to write GPS exif data to an image
    """

    def get_meta_data(
        self, image_path: str
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[datetime.datetime],
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        """
        Method allowing to extract meta information of an image containing:
        - GPS latitude, longitude and altitude
        - Timestamp of image
        - Camera manufacturer
        - Camera type
        - User comment
        :param image_path:
        :return: Tuple of (lat, lng, alt, timestamp, camera_manufacturer, camera, comment)
        """
        image = Image.open(image_path)
        exif = image.getexif()
        gps_info = exif.get_ifd(34853)
        if gps_info is None:
            lat = None
            lng = None
            alt = None
        else:
            gps_exif = {GPSTAGS.get(key, key): value for key, value in gps_info.items()}
            gps_latitude = gps_exif.get("GPSLatitude")
            gps_longitude = gps_exif.get("GPSLongitude")
            gps_latitude_ref = gps_exif.get("GPSLatitudeRef") or "N"
            gps_longitude_ref = gps_exif.get("GPSLongitudeRef") or "E"
            if gps_latitude is not None:
                lat = get_decimal_from_dms(gps_latitude, gps_latitude_ref)
            else:
                lat = None
            if gps_longitude is not None:
                lng = get_decimal_from_dms(gps_longitude, gps_longitude_ref)
            else:
                lng = None
            alt = gps_exif.get("GPSAltitude") or 0

        timestamp = exif.get(306)

        if timestamp is not None:
            timestamp = datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")

        camera_manufacturer = exif.get(271)
        camera = exif.get(272)
        comment = exif.get(37510)
        return lat, lng, alt, timestamp, camera_manufacturer, camera, comment

    def get_timestamp(self, image_path: str) -> Optional[datetime.datetime]:
        """
        Read timestamp of image
        :param image_path: from which EXIF should be read
        :return: timestamp if available
        """
        return self.get_meta_data(image_path)[3]

    def get_camera(self, image_path: str) -> Tuple[str, str]:
        """
        Reads camera information of image
        :param image_path: from which EXIF should be read
        :return: Tuple describing (camera_manufacturer, camera_type) if available
        """
        return self.get_meta_data(image_path)[4:6]

    def get_user_comment(self, image_path: str) -> Optional[str]:
        """
        Reads user comment of image
        :param image_path: from which EXIF should be read
        :return: user comment if available
        """
        return self.get_meta_data(image_path)[6]

    def get_gps(self, image_path: str) -> Tuple[float, float, float]:
        """
        Read gps EXIF information from image
        :param image_path: from which EXIF should be read
        :return: (lat, lng, alt) tuple
        """
        return self.get_meta_data(image_path)[0:3]

    def write_gps(
        self,
        image_path: str,
        lng: float,
        lat: float,
        alt: float,
        camera_manufacturer: str = "",
        camera: str = "",
        timestamp: Optional[datetime.datetime] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Method for writing GPS exif information to an image
        :param image_path: to which exif should be added
        :param lng: longitude
        :param lat: latitude
        :param alt: altitude
        :param camera_manufacturer: Manufacturer of the camera (Required for tools such as https://www.pic2map.com/!)
        :param camera: Camera used (Required for tools such as https://www.pic2map.com/!)
        :param timestamp: Timestamp, when image/video frame was created
        :param comment: Additional comment allowing to add text payload such as milliseconds after video start (e.g. AirDataFrame#time)
        :return: None
        """
        image = Image.open(image_path)
        exif = image.getexif()
        gps_value = {
            0: b"\x02\x03\x00\x00",
            1: "N",
            2: get_dms_from_decimal(lat),
            3: "E",
            4: get_dms_from_decimal(lng),
            5: b"\x00",
            6: alt,
            9: "A",
            18: "WGS-84\x00",
        }
        exif[34853] = gps_value
        exif[271] = camera_manufacturer
        exif[272] = camera
        if timestamp is not None:
            timestamp_str = timestamp.strftime("%Y:%m:%d %H:%M:%S")
            # '2022:10:05 10:08:42'
            exif[306] = timestamp_str
            exif[36867] = timestamp_str
            exif[36868] = timestamp_str
        if comment is not None:
            exif[37510] = comment
        image.save(image_path, exif=exif)
