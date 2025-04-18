import json
import os
import shutil
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

class ProjectionType(Enum):
    NoProjection = 0 # use if no projection should be applied for detection
    OrthographicProjection = 1 # use if only orthographic projection should be applied
    AlfsProjection = 2 # use if light field rendering should be applied

if __name__ == '__main__':
    # Define steps to do
    steps_to_do = {
        "extract_frames": False, # if frames are already available from previous export, set to false
        "project_frames": True, # if frames are already projected (or you don't want to project them at all), set to false
        "skip_existing_projection": True, # if a projection is already available skip this individual one
        "projection_method": ProjectionType.AlfsProjection, # define the projection style that should be used (this also determines, which files are used for the detection!)
        "detect_animals": False # flag if wildlife detection should be executed after data preparation
    }

    # St. Pankraz is available as testdata set (c.f. folder /alfs_detection/testdata/stpankraz)
    # However, the videos have to be downloaded separately (see /alfs_detection/testdata/stpankraz/todo.txt)

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
    camera_name = "T" # Name of the camera, either T or W

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
    if steps_to_do["extract_frames"]:
        shutil.rmtree(target_folder, ignore_errors=True)
        os.makedirs(target_folder, exist_ok=True)
        print("1. Extracting frames")

        # prepare all input files for the timed pose extractor
        with open(path_to_dem_json, "r") as f:
            dem_json = json.load(f)
            origin = dem_json["origin_wgs84"]

        ad_origin = AirDataFrame()
        ad_origin.latitude = origin["latitude"]
        ad_origin.longitude = origin["longitude"]
        ad_origin.altitude = origin["altitude"]

        with open(path_to_calibration) as f:
            calibration_res = json.load(f)

        # prepare the required objects for extracting the video frames
        accessor = CalibratedVideoFrameAccessor(calibration_res)
        extractor = TimedPoseExtractor(
            accessor, camera_name=Camera.from_string(camera_name)
        )

        # now lets start the hard video frame mining
        extractor.extract(
            target_folder, air_data_path, videos, srts, origin=ad_origin, include_gps=True
        )
    else:
        print("1. Skipping frame extraction")

    #####################################################################################################################################################################

    # Step 2: Access frames
    # get the poses file we will need it multiple times
    with open(os.path.join(target_folder, "poses.json"), "r") as f:
        poses = json.load(f)
    frame_count = len(poses["images"])

    if steps_to_do["project_frames"] and steps_to_do["projection_method"] != ProjectionType.NoProjection:
        print("2. Starting projection")
        # get all the flight correction data
        with open(path_to_flight_correction, 'r') as file:
            correction = json.load(file)
        translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
        cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
        rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
        correction = Transform(cor_translation, Quaternion.from_eulers(cor_rotation_eulers))

        # prepare some variables for later releasing resources
        ctx = None
        mask_shot = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
            # initialize the ModernGL context
            ctx = make_mgl_context()

            # load digital elevation model
            mesh_data, texture_data = read_gltf(path_to_dem)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_aabb = get_aabb(mesh_data.vertices)

            # prepare the mask file
            mask_shot = CtxShot._cvt_img(cv2.imread(os.path.join(target_folder, f"mask_{camera_name}.png"), cv2.IMREAD_UNCHANGED))
            mask = TextureData(mask_shot)

            # prepare the camera settings
            input_resolution = Resolution(INPUT_WIDTH, INPUT_HEIGHT)
            render_resolution = Resolution(RENDER_WIDTH, RENDER_HEIGHT)
            settings = BaseSettings(
                count=frame_count, initial_skip=0, add_background=False,
                camera_position_mode=CameraPositioningMode.FirstShot, fovy=FOVY, aspect_ratio=ASPECT_RATIO, orthogonal=True,
                ortho_size=(ORTHO_WIDTH, ORTHO_HEIGHT), correction=correction, resolution=render_resolution
            )

            alfs_rendering = steps_to_do["projection_method"] == ProjectionType.AlfsProjection

            start_idx = alfs_number_of_neighbors if alfs_rendering else 0
            total_indices = frame_count - start_idx
            number_of_renderings = (total_indices + (sample_rate - 1)) // sample_rate
            cnt = 1
            # now it is time to project the video frames
            for imagefile_idx in range(0, frame_count):
                if alfs_rendering and imagefile_idx < alfs_number_of_neighbors:
                    # skip the first x frames if ALFS should be applied since there is no "negative neighborhood" only positive people around ;)
                    continue

                if imagefile_idx % sample_rate != 0:
                    # skip some frames based on the sampling rate
                    continue

                # get the image related information from the poses file
                image_metadata = poses["images"][imagefile_idx]
                image = os.path.join(target_folder, image_metadata["imagefile"])
                print(f"{cnt} / {number_of_renderings}: Rendering image {image}")
                cnt += 1
                if not os.path.exists(image):
                    # if source image is for whatever reason not available skip it
                    print(f"Input image not available. Skip it. {image}")
                    continue

                if not alfs_rendering:
                    save_name = image.replace(".", "_projected.")
                else:
                    save_name = image.replace(".", "_alfs.")

                if steps_to_do["skip_existing_projection"] and os.path.exists(save_name):
                    print(f"Already projected. Skip it. {image}")
                    continue

                # prepare some additional variables that have to be released
                shot = None
                renderer = None
                try:
                    # get the camera extrinsics
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

                    # time to prepare the rendering
                    shot = create_shot(image, image_metadata, ctx, correction)
                    single_shot_camera = make_camera(mesh_aabb, [shot], settings, rotation=Quaternion.from_eulers(
                        [(eulers[0] - cor_rotation_eulers[0]), (eulers[1] - cor_rotation_eulers[1]),
                         (eulers[2] - cor_rotation_eulers[2])]))
                    renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)

                    # now it is time to render our orthographic projections or light fields
                    if not alfs_rendering:
                        # we only want to render one image, so not too much to do
                        shot_loader = make_shot_loader([shot])

                        # time to render
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
                    else:
                        # for light fields we also have to get the neighboring frames before and after our central image
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

                        # together with the neighbors it is time to render
                        shot_loader = make_shot_loader(shots_before + [shot] + shots_after)
                        renderer.render_integral(shot_loader,
                                                 mask=mask,
                                                 save=True,
                                                 release_shots=True,
                                                 save_name=save_name)
                        print(f"Rendered: {save_name}")
                finally:
                    # free up resources
                    print()
                    release_all(renderer)
                    print()
        finally:
            # free up resources
            release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
    else:
        print("2. Skipping projection")

    #####################################################################################################################################################################
    # 3. Do wildlife detection
    if steps_to_do["detect_animals"]:
        print("3. Starting wildlife detection")
        # now the final step has arrived: the inference of our AI models, so load it in the first glance
        m = UltralyticsYoloDetector(model_name=model_name, min_confidence=min_confidence)
        bb_writer = YoloWriter()

        # todo remove in future: just for testing
        # skip = True

        # now lets go over all images and use them for inference
        for imagefile_idx in range(0, frame_count):
            if steps_to_do[
                "projection_method"] == ProjectionType.AlfsProjection and imagefile_idx < alfs_number_of_neighbors:
                # skip the first x frames if ALFS was applied since we don't have them rendered
                continue

            if imagefile_idx % sample_rate != 0:
                # skip some frames based on the sampling rate
                continue

            # get the image related information from the poses file
            image_metadata = poses["images"][imagefile_idx]
            image = os.path.join(target_folder, image_metadata["imagefile"])

            # todo remove in future: just for testing
            # if image.endswith("5345-5464-5464.jpg"):
            #     skip = False
            #
            # if skip:
            #     continue

            # however if we want to use the AI models on the orthographic projections or light fields, we have to adapt the file name
            if steps_to_do["projection_method"] == ProjectionType.OrthographicProjection:
                image = image.replace(".", "_projected.")
            elif steps_to_do["projection_method"] == ProjectionType.AlfsProjection:
                image = image.replace(".", "_alfs.")

            # if file can't be found for whatever reason, skip it
            if not os.path.exists(image):
                print(f"Input image not available. Skip it. {image}")
                continue

            # now it is time to get all the bounding boxes, since we are rendering in a bigger resolution compared to our input resolution,
            # we create tiles that are used for inference, but the detected objects are stitched together again
            # (probably requires some NMS as post-processing, but this is a future problem).
            bounding_boxes = []
            current_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            for tile in tile_image(current_image, INPUT_WIDTH):
                boxes = m.detect_frame(imagefile_idx, tile[2])
                for box in boxes:
                    # add the offset from the tiling
                    box.start_x += tile[0]
                    box.end_x += tile[0]
                    box.start_y += tile[1]
                    box.end_y += tile[1]
                    bounding_boxes.append(box)
            # now it is time to write the bounding boxes to the disk
            bb_writer.write_boxes(target_folder, m.get_labels(), [(Path(image).stem, current_image, bounding_boxes)])
            # todo remove in future: just for testing
            # break
    else:
        print("3. Skipping wildlife detection")
