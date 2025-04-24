import json
import time
from pathlib import Path
import contextily as ctx

import geopandas as gpd
import matplotlib.pyplot as plt
import imageio
import os
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

if __name__ == '__main__':
    # Script to create a video visualization of the monitoring process created with the bambi_detection.py script
    # Requires at least "export_flight_data": True and "export_individual_polygons": True
    # May also consider projected frames based on "project_labels": True

    # Set paths
    input_path = r"..\..\testdata\stpankraz\target"
    output_folder = r"..\..\testdata\stpankraz\target\visualization"
    video_output = os.path.join(output_folder, "flight_visualization.mp4")
    poses_file = os.path.join(input_path, "poses.json")
    map_dpi = 300 # DPI of the map visualization, used to determine the pixel resolution
    local_crs = 32633
    include_base_map = False
    alfs_rendering = False

    ####################################################################################################################

    if not os.path.exists(input_path):
        print("Input path does not exist. Run bambi_detection.py first.")
        exit()

    with open(poses_file, 'r') as f:
        poses = json.load(f)

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize combined GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(geometry=[])
    gps_points = []
    previous_labels = gpd.GeoDataFrame(geometry=[])

    # For video creation
    screenshot_files = []
    images = []

    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.3, label='Static Polygons'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.5, label='Previous Areas'),
        Patch(facecolor='blue', edgecolor='black', alpha=1.0, label='Current Area'),
        Line2D([0], [0], marker='o', color='w', label='GPS Points',
               markerfacecolor='red', markersize=8),
        Line2D([0], [0], color='green', lw=2, label='Path')
    ]

    final_area = gpd.read_file(os.path.join(input_path, "area.geojson"))#.to_crs(local_crs)

    width = None
    height = None
    # Step-by-step combination and plotting
    starttime = time.time()
    print("Creating video frames")
    for idx, image_metadata in enumerate(poses["images"]):
        geojson_file = os.path.join(input_path, f"{idx}_area.geojson")
        if not os.path.exists(geojson_file):
            continue
        print(f"Processing image {idx}")
        image = image_metadata["imagefile"]
        if alfs_rendering:
            image = Path(image).stem + "_alfs" + Path(image).suffix
        else:
            image = Path(image).stem + "_projected" + Path(image).suffix
        image_path = os.path.join(input_path, image)

        if not os.path.exists(image_path):
            print(f"Could find rendering at {image_path}")
            continue

        images.append(image_path)
        lng = image_metadata["lng"]
        lat = image_metadata["lat"]
        gps_point = Point(lng, lat)
        gps_points.append(gps_point)

        labels = None
        projected_labels = os.path.join(input_path, Path(image).stem + "_projected.geojson")
        if os.path.exists(projected_labels):
            labels = projected_labels
        else:
            alfs_labels = os.path.join(input_path, Path(image).stem + "_alfs.geojson")
            if os.path.exists(alfs_labels):
                labels = alfs_labels

        gdf = gpd.read_file(geojson_file)#.to_crs(local_crs)

        # Combine with current
        combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)
        combined_gdf = combined_gdf.dissolve()  # Optional: dissolve to one geometry

        # Create GeoDataFrame for GPS path
        if len(gps_points) > 1:
            path_line = LineString(gps_points)
            path_gdf = gpd.GeoDataFrame(geometry=[path_line], crs=4326)#.to_crs(local_crs)
        else:
            path_gdf = None

        if width is None and height is None:
            read_image = imageio.v2.imread(image_path)
            width = read_image.shape[1]
            height = read_image.shape[0]

        # Plot the combined polygon
        fig, ax = plt.subplots(figsize=(width / map_dpi, height / map_dpi))
        if len(previous_labels) > 0:
            previous_labels.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Previous detections')
        if labels is not None:
            label_gdf = gpd.read_file(labels)#.to_crs(local_crs)
            label_gdf.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Current detections')
            previous_labels = pd.concat([previous_labels, label_gdf], ignore_index=True)

        # Highlight current GPS point
        gpd.GeoDataFrame(geometry=[gps_point]).plot(ax=ax, color='yellow', markersize=10, edgecolor='black', marker='o')

        combined_gdf.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Combined Area')
        if path_gdf is not None:
            path_gdf.plot(ax=ax, color='green', linewidth=2, label='Path')

        final_area.plot(ax=ax, color='white', edgecolor='black', alpha=0.1, label='Area')
        if include_base_map:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=16)
        # ax.set_title(f'Step {idx + 1}: Combined Area + GPS')
        ax.axis('off')
        # ax.legend(handles=legend_elements, loc='best')

        screenshot_path = os.path.join(output_folder, f'step_{idx + 1}.png')
        plt.savefig(screenshot_path, dpi=map_dpi)
        plt.close(fig)

        screenshot_files.append(screenshot_path)
    print(f"Created video frames in {time.time() - starttime} seconds")
    starttime = time.time()
    print("Creating video")
    # Create video from screenshots
    with imageio.get_writer(video_output, mode='I', fps=1) as writer:
        for image, screenshot in zip(images, screenshot_files):
            i1 = imageio.v2.imread(image)
            i2 = imageio.v2.imread(screenshot)[:, :, :3]
            writer.append_data(np.hstack((i1, i2)))
    print(f"Created video in {time.time() - starttime} seconds")
    for screenshot in screenshot_files:
        os.remove(screenshot)

    print(f"Video created at: {video_output}")