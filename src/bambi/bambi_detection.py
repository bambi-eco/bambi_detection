import json
import math
import os
import shutil
from pathlib import Path

import cv2
from alfspy.core.geo import Transform
from alfspy.core.rendering import Resolution, Renderer, RenderResultMode, TextureData
from alfspy.core.util.geo import get_aabb
from alfspy.orthografic_projection import get_camera_for_frame
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import read_gltf, process_render_data, make_mgl_context, make_camera, make_shot_loader, \
    release_all
from pyproj.enums import TransformDirection

from bambi.ai.models.ultralytics_yolo_detector import UltralyticsYoloDetector
from bambi.ai.output.yolo_writer import YoloWriter

from bambi.airdata.air_data_frame import AirDataFrame
from bambi.domain.camera import Camera
from bambi.video.calibrated_video_frame_accessor import CalibratedVideoFrameAccessor
from bambi.webgl.timed_pose_extractor import TimedPoseExtractor

from pyproj import CRS, Transformer
from pyrr import Quaternion
from trimesh import Trimesh
from bambi.util.projection_util import *
import webcolors

if __name__ == '__main__':
    # Define steps to do
    steps_to_do = {
        "extract_frames": True, # if frames are already available from previous export, set to false, otherwise it will also delete existing exports!
        "project_frames": True, # if frames are already projected (or you don't want to project them at all), set to false
        "skip_existing_projection": True, # if a projection is already available skip this individual one
        "projection_method": ProjectionType.OrthographicProjection, # define the projection style that should be used (this also determines, which files are used for the detection!)
        "detect_animals": True, # flag if wildlife detection should be executed after data preparation
        "project_labels": True, # flag if detected wildlife labels should be projected based on the digital elevation model
        "export_flight_data": True  # if flight relevant data should be exported like the route and the monitored area, as well as statistics about the area in m² and the perimeter in m. Be aware that this is affected by the selected sample_rate.
    }

    # St. Pankraz is available as testdata set (c.f. folder /alfs_detection/testdata/stpankraz)
    # However, the videos have to be downloaded separately (see /alfs_detection/testdata/stpankraz/todo.txt)

    # Define input data
    # Sorted paths to all videos of the current flight
    videos = [
        r"C:\Users\P41743\Desktop\stpankraz\DJI_20240822074151_0001_T_point0.MP4",
        r"C:\Users\P41743\Desktop\stpankraz\DJI_20240822074745_0002_T_point0.MP4"
    ]

    # Sorted paths to all SRT files of the current flight
    srts = [srt.replace(".MP4", ".SRT") for srt in videos]

    # Path to the AirData log of the current flight
    air_data_path = r"C:\Users\P41743\Desktop\stpankraz\air_data.csv"

    # Target folder, where all the output of defined steps is put in (is also used as input path for subsequent steps e.g. for Project_Frames after Extract_Frames)
    target_folder = r"C:\Users\P41743\Desktop\stpankraz\target"

    # Path to the GLTF based DEM file
    path_to_dem = r"C:\Users\P41743\Desktop\stpankraz\dem_mesh_r2.gltf"

    # Path to the metadata for the DEM file
    path_to_dem_json = path_to_dem.replace(".gltf", ".json")

    # Path to the file containing the camera calibration
    path_to_calibration = r"C:\Users\P41743\Desktop\stpankraz\T_calib.json"

    # Path to the file containing flight specific corrections
    path_to_flight_correction = r"C:\Users\P41743\Desktop\stpankraz\correction.json"

    # Name of the camera, either T or W
    camera_name = "T"

    target_crs = CRS.from_epsg(32633) # make sure that your DEM is matching the target_crs!

    # Define rendering settings
    sample_rate = 1 # sample rate of frames that are considered for projection, animal detection and export of flight data (won't affect the first step with the general frame-extraction)
    alfs_number_of_neighbors = 100 # number of neighbors before, as well as after the central frame of an light field (results in a light field based on n + 1 + n images)
    alfs_neighbor_sample_rate = 10 # sample rate of the neighbors

    ORTHO_WIDTH = 70 # orthographic width of the rendered image (in m)
    ORTHO_HEIGHT = 70 # orthographic height of the rendered image (in m)
    RENDER_WIDTH = 2048 # pixel width of the rendered image
    RENDER_HEIGHT = 2048 # pixel height of the rendered image
    ADD_BACKGROUND = False # flag if the texture of the DEM should be included as background to the projected image
    FOVY = 50 # field of view of the rendering camera used for projection
    ASPECT_RATIO = 1 # aspect ratio of the rendering camera used for projection

    # Define yolo model settings
    model_name = "yolov11-20250326" # model that should be used for the wildlife detection
    min_confidence = 0.5 # minimum confidence that should be considered by the model

    #####################################################################################################################################################################
    #####################################################################################################################################################################
    #####################################################################################################################################################################
    input_crs = CRS.from_epsg(4326) # WGS 84 coordinates. Don't change since GeoJSON won't work with another CRS.

    # Step 1: Extract frames
    rel_transformer = Transformer.from_crs(input_crs, target_crs)
    with open(path_to_dem_json, "r") as f:
        dem_json = json.load(f)
    x_offset = dem_json["origin"][0]
    y_offset = dem_json["origin"][1]
    z_offset = dem_json["origin"][2]

    if steps_to_do["extract_frames"]:
        shutil.rmtree(target_folder, ignore_errors=True)
        os.makedirs(target_folder, exist_ok=True)
        print("1. Extracting frames")

        # prepare all input files for the timed pose extractor

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
            accessor,
            rel_transformer=rel_transformer,
            camera_name=Camera.from_string(camera_name)
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

    # get all the flight correction data
    with open(path_to_flight_correction, 'r') as file:
        correction = json.load(file)
    translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
    cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
    rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
    correction = Transform(cor_translation, Quaternion.from_eulers(cor_rotation_eulers))

    mask_shot = CtxShot._cvt_img(
        cv2.imread(os.path.join(target_folder, f"mask_{camera_name}.png"), cv2.IMREAD_UNCHANGED))

    mask_height, mask_width, _ = mask_shot.shape

    # prepare the camera settings
    input_resolution = Resolution(mask_width, mask_height)
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

    if steps_to_do["project_frames"] and steps_to_do["projection_method"] != ProjectionType.NoProjection:
        print("2. Starting projection")
        # prepare some variables for later releasing resources
        ctx = None
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
            mask = TextureData(mask_shot)


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
    m = UltralyticsYoloDetector(model_name=model_name, min_confidence=min_confidence)
    if steps_to_do["detect_animals"]:
        print("3. Starting wildlife detection")
        # now the final step has arrived: the inference of our AI models, so load it in the first glance
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
            for tile in tile_image(current_image, mask_width):
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

    #####################################################################################################################################################################
    # 4. Project labels
    if steps_to_do["project_labels"] and steps_to_do["projection_method"] != ProjectionType.NoProjection:
        print("4. Projecting labels")
        colors = webcolors.names()
        ctx = None
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
                image_file_name = Path(image_metadata["imagefile"]).stem
                labels_file = os.path.join(target_folder, image_file_name + ".txt")
                labels_target_file = os.path.join(target_folder, image_file_name + ".json")
                labels_geojson_target_file = os.path.join(target_folder, image_file_name + ".geojson")
                if os.path.exists(labels_file):
                    print(f"Projecting labels file {image_file_name}.txt")
                    frame_labels = []
                    with open(labels_file, 'r') as f:
                        for line in f:
                            # Get the yolo based labels
                            values = line.strip().split()
                            if len(values) == 5:
                                class_id = int(values[0])
                                x_center = float(values[1])
                                y_center = float(values[2])
                                width = float(values[3])
                                height = float(values[4])

                                # Convert to pixel coordinates
                                img_height, img_width = input_resolution.height, input_resolution.width
                                x1 = int((x_center - width / 2) * img_width)
                                y1 = int((y_center - height / 2) * img_height)
                                x2 = int((x_center + width / 2) * img_width)
                                y2 = int((y_center + height / 2) * img_height)

                                frame_labels.append((class_id, [x1, y1, x2, y1, x2, y2, x1, y2]))

                    if len(frame_labels) > 0:
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

                        # project labels
                        camera = get_camera_for_frame(poses, imagefile_idx, cor_rotation_eulers, cor_translation)
                        projected_labels = []
                        points = []
                        for class_id, poly_coords in frame_labels:
                            class_name = m._labels[class_id]
                            world_coordinates = label_to_world_coordinates(poly_coords, input_resolution, tri_mesh, camera)

                            # transform the labels to WGS84 coordinates and export it as GeoJSON
                            xx = world_coordinates[:,0] + x_offset
                            yy = world_coordinates[:,1] + y_offset
                            zz = world_coordinates[:,2] + z_offset
                            transformed = rel_transformer.transform(xx, yy, zz, direction=TransformDirection.INVERSE)
                            points.append({
                                "type": "Feature",
                                "properties": {
                                    "title": class_name,
                                    "className": class_name,
                                    "marker-color": webcolors.name_to_hex(colors[class_id]),
                                    "frameIdx": imagefile_idx
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                      list(zip(transformed[1], transformed[0])) + [[transformed[1][0], transformed[0][0]]]
                                    ]
                                  }
                            })
                            projected_labels.append({"Class": class_name,
                                                     "DemCoordinates": world_coordinates.tolist(),
                                                     "WGS84Coordinates": list(zip(transformed[1], transformed[0], transformed[2]))})

                        with open(labels_target_file, "w") as f:
                            json.dump({
                                "Labels": projected_labels,
                                "EPSG": target_crs.srs
                            }, f)

                        with open(labels_geojson_target_file, "w") as f:
                            geo_json = {
                                "type": "FeatureCollection",
                                "features": points
                            }
                            json.dump(geo_json, f)

        finally:
            # free up resources
            release_all(ctx)
            del mesh_data
            del texture_data
            del tri_mesh
    else:
        print("4. Skipping label projection")

    #####################################################################################################################################################################
    # 5. Export of flight data
    def get_extrinsics_from_image_metdata(image_metadata):
        rotation = image_metadata["rotation"]
        rotation = [val % 360.0 for val in rotation]
        rot_len = len(rotation)
        if rot_len == 3:
            eulers = [np.deg2rad(val) for val in rotation]
        else:
            raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')
        R_mat, _ = cv2.Rodrigues(np.array(eulers))

        # build 4×4 extrinsics
        extrinsic = np.eye(4, dtype=float)
        extrinsic[:3, :3] = R_mat
        extrinsic[:3, 3] = np.array(image_metadata["location"])
        return extrinsic

    if steps_to_do["export_flight_data"]:
        print("5. Export of flight data (route and monitored area)")
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
            # load digital elevation model
            mesh_data, texture_data = read_gltf(path_to_dem)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)

            # now it is time to project the video frames
            cnt = 1

            route = []
            extrinsics = []
            for imagefile_idx in range(0, frame_count):
                if alfs_rendering and imagefile_idx < alfs_number_of_neighbors:
                    # skip the first x frames if ALFS should be applied since there is no "negative neighborhood" only positive people around ;)
                    continue

                if imagefile_idx % sample_rate != 0:
                    # skip some frames based on the sampling rate
                    continue

                image_metadata = poses["images"][imagefile_idx]
                image = os.path.join(target_folder, image_metadata["imagefile"])
                route.append((image_metadata["lng"], image_metadata["lat"]))
                cnt += 1
                if not os.path.exists(image):
                    # if source image is for whatever reason not available skip it
                    print(f"Input image not available. Skip it. {image}")
                    continue

                # get the extrinsics of the central frame
                extrinsics.append(get_extrinsics_from_image_metdata(image_metadata))
                if alfs_rendering:
                    # get the extrinsics for neighboring frames
                    for image_before_idx in range(0, alfs_number_of_neighbors):
                        if image_before_idx % alfs_neighbor_sample_rate == 0:
                            idx = imagefile_idx - alfs_number_of_neighbors + image_before_idx
                            image_before_metadata = poses["images"][idx]
                            extrinsics.append(get_extrinsics_from_image_metdata(image_before_metadata))

                    for image_after_idx in range(1, alfs_number_of_neighbors + 1):
                        if image_after_idx % alfs_neighbor_sample_rate == 0:
                            idx = imagefile_idx + image_after_idx
                            image_after_metadata = poses["images"][idx]
                            extrinsics.append(get_extrinsics_from_image_metdata(image_after_metadata))

            # prepare the camera intrinsics
            fov_y = poses["images"][0]["fovy"][0]
            fy = (mask_height / 2) / np.tan(math.radians(fov_y) / 2)
            cx = mask_width / 2
            cy = mask_height / 2
            intrinsics = np.array([[fy, 0, cx], [0, fy, cy], [0, 0, 1]])

            # do the calculation
            area, perimeter, final_coordinates = measure_area(tri_mesh, intrinsics, extrinsics, mask_shot, rel_transformer, x_offset, y_offset, z_offset, 4)
            print(f"Observed area: {area} m² with a perimeter of {perimeter}m")

            # export the flight route based on the poses.json
            with open(os.path.join(target_folder, f"route.json"), "w") as f:
                json.dump({
                      "type": "FeatureCollection",
                      "features": [
                          {
                              "type": "Feature",
                              "geometry": {
                                  "type": "LineString",
                                  "coordinates": route
                              },
                              "properties": {}
                          }
                      ]
                  }, f)

            # export the projected area and at the measurments as metadata
            with open(os.path.join(target_folder, f"area.json"), "w") as f:
                json.dump({
                                "type": "FeatureCollection",
                                "features": [
                                    {
                                        "type": "Feature",
                                        "properties": {
                                            "area": area,
                                            "perimeter": perimeter,
                                            "marker-color": "#ff0000"
                                        },
                                        "geometry": {
                                            "type": "Polygon",
                                            "coordinates": [
                                                final_coordinates
                                            ]
                                        }
                                    }
                                ]
                            }, f)

        finally:
            del mesh_data
            del texture_data
            del tri_mesh
    else:
        print("5. Skipping export of flight data")