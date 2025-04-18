from typing import Generator, List, Optional, Union

from bambi.domain.camera import Camera
from bambi.domain.drone import Drone
from bambi.srt.srt_frame import SrtFrame


class SrtWriter:
    """
    Class allowing to create SRT files based on SRT frames
    """

    def write(
        self,
        target_file: str,
        frames: Union[Generator[SrtFrame, None, None], List[SrtFrame]],
        drone_type: Drone,
        camera: Camera,
    ) -> None:
        """
        Method for writing frames to a target file
        :param target_file: target file
        :param frames: to be written
        :param drone_type: drone for which SRT file should be created
        :param camera: camera for which SRT file should be created (only relevant if drone_type == M30)
        :return: None
        """
        if not Drone.is_camera_supported(drone_type, camera):
            raise Exception(
                f"Given camera {camera} is not supported by drone {drone_type}"
            )

        font_size = 36 if drone_type == Drone.M2EA else 28

        with open(target_file, "w+") as file:
            for idx, input_frame in enumerate(frames):
                frame = SrtFrame()
                frame.__dict__ = input_frame.__dict__.copy()
                for key, value in frame.__dict__.items():
                    if value is None:
                        frame.__dict__[key] = 0

                file.write(f"{idx+1}\n")
                file.write(
                    f"{frame.start.strftime('%H:%M:%S,%f')} --> {frame.end.strftime('%H:%M:%S,%f')}\n"
                )
                file.write(
                    f'<font size="{font_size}">FrameCnt: {frame.FrameCnt}, DiffTime: {frame.DiffTime}\n'
                )
                file.write(f"{frame.timestamp}\n")
                if drone_type == Drone.M2EA:
                    file.write(f"[color_md : {frame.color_md}] ")
                    file.write(f"[latitude: {frame.latitude}] ")
                    file.write(f"[longtitude: {frame.longitude}] ")
                    file.write(f"[rel_alt: {frame.rel_alt} abs_alt: {frame.abs_alt}] ")
                    file.write(
                        f"[Drone: Yaw:{frame.drone_yaw}, Pitch:{frame.drone_pitch}, Roll:{frame.drone_roll}] "
                    )
                    file.write(f"</font>\n\n")
                elif drone_type in [Drone.M30T, Drone.M3T]:
                    if camera == Camera.Zoom or (
                        drone_type == Drone.M30T and camera == Camera.Wide
                    ):
                        file.write(f"[iso: {frame.iso}] ")
                        file.write(f"[shutter: {frame.shutter}] ")
                        file.write(f"[fnum: {frame.fnum}] ")
                        file.write(f"[ev: {frame.ev}] ")
                        file.write(f"[color_md: {frame.color_md}] ")
                        file.write(f"[ae_meter_md: {frame.ae_meter_md}] ")
                    file.write(f"[focal_len: {frame.focal_len}] ")
                    file.write(f"[dzoom_ratio: {frame.dzoom_ratio}], ")
                    file.write(f"[latitude: {frame.latitude}] ")
                    file.write(f"[longitude: {frame.longitude}] ")
                    file.write(f"[rel_alt: {frame.rel_alt} abs_alt: {frame.abs_alt}] ")
                    file.write(
                        f"[gb_yaw: {frame.gb_yaw} gb_pitch: {frame.gb_pitch} gb_roll: {frame.gb_roll}"
                    )
                    if camera != "s":
                        file.write("] ")
                    file.write(f"</font>\n\n")
                elif drone_type == Drone.M300:
                    file.write(f"[iso: {frame.iso}] ")
                    file.write(f"[shutter: {frame.shutter}] ")
                    file.write(f"[fnum: {frame.fnum}] ")
                    file.write(f"[ev: {frame.ev}] ")
                    file.write(f"[focal_len: {frame.focal_len}] ")
                    file.write(f"[dzoom: {frame.dzoom}]\n")
                    file.write(f"[latitude: {frame.latitude}] ")
                    file.write(f"[longitude: {frame.longitude}] ")
                    file.write(f"[rel_alt: {frame.rel_alt} abs_alt: {frame.abs_alt}]\n")
                    file.write(
                        f"[drone_speedx: {frame.drone_speedx} drone_speedy: {frame.drone_speedy} drone_speedz: {frame.drone_speedz}]\n"
                    )
                    file.write(
                        f"[drone_yaw: {frame.drone_yaw} drone_pitch: {frame.drone_pitch} drone_roll: {frame.drone_roll}]\n"
                    )
                    file.write(
                        f"[gb_yaw: {frame.gb_yaw} gb_pitch: {frame.gb_pitch} gb_roll: {frame.gb_roll}]\n\n"
                    )
                    file.write(f"0\n")
                    file.write(
                        f"[ae_meter_md : {frame.ae_meter_md}] [dzoom_ratio: {frame.dzoom_ratio}, delta:{frame.delta}] [color_md : {frame.color_md}] [ct : {frame.ct}]\n"
                    )
                    file.write(f"</font>\n\n")
