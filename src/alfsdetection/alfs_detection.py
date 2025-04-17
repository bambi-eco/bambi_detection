import json
import os
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from alfspy.core.geo import Transform
from alfspy.core.rendering import CtxShot, Resolution, Renderer, RenderResultMode, TextureData
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import read_gltf, process_render_data, make_mgl_context, make_camera, make_shot_loader, \
    release_all
from bambi.ai.models.ultralytics_yolo_detector import UltralyticsYoloDetector
from bambi.ai.output.yolo_writer import YoloWriter
from bambi.pipeline.airdata.air_data_frame import AirDataFrame
from bambi.pipeline.domain.camera import Camera
from bambi.pipeline.domain.drone import Drone
from bambi.pipeline.domain.sensor import SensorResolution
from bambi.pipeline.video.calibrated_video_frame_accessor import CalibratedVideoFrameAccessor
from bambi.pipeline.webgl.timed_pose_extractor import TimedPoseExtractor
from pyrr import Vector3, Quaternion
from trimesh import Trimesh


class ProjectionType(Enum):
    NoProjection = 0 # use if no projection should be applied for detection
    OrthographicProjection = 1 # use if only orthographic projection should be applied
    AlfsProjection = 2 # use if light field rendering should be applied

def create_shot(image, image_metadata, ctx, correction):
    position = Vector3(image_metadata["location"])
    rotation = image_metadata["rotation"]
    rotation = [val % 360.0 for val in rotation]
    rot_len = len(rotation)
    if rot_len == 3:
        eulers = [np.deg2rad(val) for val in rotation]
        rotation = quaternion_from_eulers(eulers, 'zyx')
    else:
        raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

    fov = image_metadata["fovy"][0]
    return CtxShot(ctx, image, position, rotation, fov, 1, correction, lazy=True)

def tile_image(img, tile_size):
    width, height, _ = img.shape

    # Calculate step size for overlapping tiles
    step_x = (width - tile_size) // 2
    step_y = (height - tile_size) // 2

    if step_x == 0 or step_y == 0:
        return [(0, 0, img)]

    # Calculate coordinates for the nine tiles
    tiles = []
    for y in [0, step_y, height - tile_size]:  # Top, middle, bottom rows
        for x in [0, step_x, width - tile_size]:  # Left, middle, right columns
            tiles.append((x, y))

    result = []
    # Create tiles and save them along with their labels
    for idx, (x, y) in enumerate(tiles):
        # Process image tile
        result.append((x, y, img[y:y + tile_size, x:x + tile_size]))
    return result


if __name__ == '__main__':
    # Define steps to do
    steps_to_do = {
        "extract_frames": False, # if frames are already available from previous export, set to false
        "project_frames": False,
        "projection_method": ProjectionType.NoProjection,
        "detect_animals": True
    }

    # Define input data
    videos = [
        r"C:\Users\P41743\Desktop\stpankraz\DJI_20240822074151_0001_T_point0.MP4",
        r"C:\Users\P41743\Desktop\stpankraz\DJI_20240822074745_0002_T_point0.MP4"
    ]
    srts = [srt.replace(".MP4", ".SRT") for srt in videos]
    air_data_path = r"C:\Users\P41743\Desktop\stpankraz\air_data.csv"
    target_folder = r"C:\Users\P41743\Desktop\stpankraz\target"
    path_to_dem = r"C:\Users\P41743\Desktop\stpankraz\dem_mesh_r2.gltf"
    path_to_dem_json = path_to_dem.replace(".gltf", ".json")
    path_to_calibration = r"C:\Users\P41743\Desktop\stpankraz\T_calib.json"
    path_to_flight_correction = r"C:\Users\P41743\Desktop\stpankraz\correction.json"
    drone_name = "M30T"
    camera_name = "T"

    # Define rendering settings
    sample_rate = 1
    alfs_number_of_neighbors = 100
    alfs_neighbor_sample_rate = 10

    ORTHO_WIDTH = 70
    ORTHO_HEIGHT = 70
    INPUT_WIDTH = 1024
    INPUT_HEIGHT = 1024
    RENDER_WIDTH = 2048
    RENDER_HEIGHT = 2048
    ADD_BACKGROUND = False
    FOVY = 50
    ASPECT_RATIO = 1

    # Define yolo model settings
    model_name = "yolov11-20250326"
    min_confidence = 0.5

    #####################################################################################################################################################################
    #####################################################################################################################################################################
    #####################################################################################################################################################################

    # Step 1: Extract frames
    os.makedirs(target_folder, exist_ok=True)

    if steps_to_do["extract_frames"]:
        print("1. Extracting frames")
        with open(path_to_dem_json, "r") as f:
            dem_json = json.load(f)
            origin = dem_json["origin_wgs84"]

        ad_origin = AirDataFrame()
        ad_origin.latitude = origin["latitude"]
        ad_origin.longitude = origin["longitude"]
        ad_origin.altitude = origin["altitude"]

        with open(path_to_calibration) as f:
            calibration_res = json.load(f)
        accessor = CalibratedVideoFrameAccessor(calibration_res)
        extractor = TimedPoseExtractor(
            accessor, camera_name=Camera.from_string(camera_name)
        )

        sr = SensorResolution(
            Drone.from_string(drone_name), Camera.from_string(camera_name)
        )
        extractor.extract(
            target_folder, air_data_path, videos, srts, origin=ad_origin, include_gps=True
        )
    else:
        print("1. Skipping frame extraction")

    #####################################################################################################################################################################

    # Step 2: Access frames
    with open(os.path.join(target_folder, "poses.json"), "r") as f:
        poses = json.load(f)
    frame_count = len(poses["images"])

    if steps_to_do["project_frames"] and steps_to_do["projection_method"] != ProjectionType.NoProjection:
        print("2. Starting projection")
        with open(path_to_flight_correction, 'r') as file:
            correction = json.load(file)

        translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
        cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

        rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
        correction = Transform(cor_translation, Quaternion.from_eulers(cor_rotation_eulers))

        ctx = None
        mask_shot = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
            time.sleep(5) # whyever but needed for trimesh and gltf loading
            mesh_data, texture_data = read_gltf(path_to_dem)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            ctx = make_mgl_context()
            mesh_aabb = get_aabb(mesh_data.vertices)
            mask_shot = CtxShot._cvt_img(cv2.imread(os.path.join(target_folder, f"mask_{camera_name}.png"), cv2.IMREAD_UNCHANGED))
            mask = TextureData(mask_shot)

            input_resolution = Resolution(INPUT_WIDTH, INPUT_HEIGHT)
            render_resolution = Resolution(RENDER_WIDTH, RENDER_HEIGHT)
            settings = BaseSettings(
                count=frame_count, initial_skip=0, add_background=False,
                camera_position_mode=CameraPositioningMode.FirstShot, fovy=FOVY, aspect_ratio=ASPECT_RATIO, orthogonal=True,
                ortho_size=(ORTHO_WIDTH, ORTHO_HEIGHT), correction=correction, resolution=render_resolution
            )

            for imagefile_idx in range(0, frame_count):
                if steps_to_do["projection_method"] == ProjectionType.AlfsProjection and imagefile_idx < alfs_number_of_neighbors:
                    continue

                if imagefile_idx % sample_rate != 0:
                    continue


                image_metadata = poses["images"][imagefile_idx]
                image = os.path.join(target_folder, image_metadata["imagefile"])
                print(f"Rendering image {image}")
                if not os.path.exists(image):
                    print(f"Input image not available. Skip it. {image}")
                    continue
                shot = None
                renderer = None
                try:
                    position = Vector3(image_metadata["location"])
                    rotation = image_metadata["rotation"]
                    rotation = [val % 360.0 for val in rotation]
                    rot_len = len(rotation)
                    if rot_len == 3:
                        eulers = [np.deg2rad(val) for val in rotation]
                        rotation = quaternion_from_eulers(eulers, 'zyx')
                    else:
                        raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

                    fov = image_metadata["fovy"][0]

                    shot = create_shot(image, image_metadata, ctx, correction)
                    single_shot_camera = make_camera(mesh_aabb, [shot], settings, rotation=Quaternion.from_eulers(
                        [(eulers[0] - cor_rotation_eulers[0]), (eulers[1] - cor_rotation_eulers[1]),
                         (eulers[2] - cor_rotation_eulers[2])]))
                    renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)

                    if steps_to_do["projection_method"] == ProjectionType.OrthographicProjection:
                        shot_loader = make_shot_loader([shot])  # Create loader for single shot
                        save_name = image.replace(".", "_projected.")
                        renderer.project_shots(
                            shot_loader,
                            RenderResultMode.ShotOnly,
                            mask=mask,
                            integral=False,
                            save=True,
                            release_shots=True,
                            save_name_iter=iter([save_name])
                        )
                        print(f"Rendered: {save_name}")
                    elif steps_to_do["projection_method"] == ProjectionType.AlfsProjection:
                        shots_before = []
                        shots_after = []
                        for image_before_idx in range(0, alfs_number_of_neighbors):
                            if image_before_idx % alfs_neighbor_sample_rate == 0:
                                idx = imagefile_idx - alfs_number_of_neighbors + image_before_idx
                                image_before_metadata = poses["images"][idx]
                                image_before = os.path.join(target_folder, image_before_metadata["imagefile"])
                                shots_before.append(create_shot(image, image_metadata, ctx, correction))

                        for image_after_idx in range(1, alfs_number_of_neighbors + 1):
                            if image_after_idx % alfs_neighbor_sample_rate == 0:
                                idx = imagefile_idx + image_after_idx
                                image_after_metadata = poses["images"][idx]
                                image_after = os.path.join(target_folder, image_before_metadata["imagefile"])
                                shots_after.append(create_shot(image, image_metadata, ctx, correction))
                        shot_loader = make_shot_loader(shots_before + [shot] + shots_after)
                        save_name = image.replace(".", "_alfs.")
                        renderer.render_integral(shot_loader,
                                                 mask=mask,
                                                 save=True,
                                                 release_shots=True,
                                                 save_name=save_name)
                        print(f"Rendered: {save_name}")
                finally:
                    release_all(renderer)
        finally:
            release_all(ctx, mask_shot)
            del mesh_data
            del texture_data
            del tri_mesh
    else:
        print("2. Skipping projection")

    #####################################################################################################################################################################
    # 3. Do wildlife detection
    if steps_to_do["detect_animals"]:
        print("3. Starting wildlife detection")
        m = UltralyticsYoloDetector(model_name=model_name, min_confidence=min_confidence)
        bb_writer = YoloWriter()

        skip = True

        for imagefile_idx in range(0, frame_count):
            if steps_to_do["projection_method"] == ProjectionType.AlfsProjection and imagefile_idx < alfs_number_of_neighbors:
                continue

            if imagefile_idx % sample_rate != 0:
                continue

            image_metadata = poses["images"][imagefile_idx]
            image = os.path.join(target_folder, image_metadata["imagefile"])

            if image.endswith("5345-5464-5464.jpg"):
                skip = False

            if skip:
                continue

            if steps_to_do["projection_method"] == ProjectionType.OrthographicProjection:
                image = image.replace(".", "_projected.")
            elif steps_to_do["projection_method"] == ProjectionType.AlfsProjection:
                image = image.replace(".", "_alfs.")

            if not os.path.exists(image):
                print(f"Input image not available. Skip it. {image}")
                continue
            bounding_boxes = []
            current_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            for tile in tile_image(current_image, INPUT_WIDTH):
                boxes = m.detect_frame(imagefile_idx, tile[2])
                for box in boxes:
                    box.start_x += tile[0]
                    box.end_x += tile[0]
                    box.start_y += tile[1]
                    box.end_y += tile[1]
                    bounding_boxes.append(box)
            bb_writer.write_boxes(target_folder, m.get_labels(), [(Path(image).stem, current_image, bounding_boxes)])
            break
    else:
        print("3. Skipping wildlife detection")
