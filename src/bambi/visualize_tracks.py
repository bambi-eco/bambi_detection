import colorsys
import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import cv2

from src.bambi.georeference_deepsort_mot import deviating_folders


def id_to_color(identifier, *, saturation=0.65, lightness=0.5):
    """
    Deterministically map any identifier (int/str/etc.) to a distinct hex color.
    Avoids float precision issues by deriving hue from hash bytes directly.
    """
    # 32-bit fraction from the hash (enough granularity for distinct hues)
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2**32  # in [0,1)

    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)  # note HLS order
    return (r * 255, g * 255, b * 255)


if __name__ == '__main__':
    tracks_base_folder = r"Z:\dets\georeferenced_tracks"
    detections_base_folder = r"Z:\dets\source"
    images_base_folder = r"Z:\sequences"
    target_base_folder = r"Z:\sequences_drawn"

    #############################
    os.makedirs(target_base_folder, exist_ok=True)

    track_files = {}
    detection_files = {}
    image_folders = {}

    for root, dirs, files in os.walk(tracks_base_folder):
        length = len(files)
        for file_idx, file in enumerate(files):
            if file.endswith(".csv") and "_" in file:
                full_file_path = os.path.join(root, file)
                p = Path(full_file_path)
                if p.stem != "26_3":
                    continue
                track_files[p.parent.name + "/" + p.stem] = full_file_path

    for root, dirs, files in os.walk(detections_base_folder):
        length = len(files)
        for file_idx, file in enumerate(files):
            full_file_path = os.path.join(root, file)
            p = Path(full_file_path)
            if p.parent.name + "/" + p.stem in track_files:
                detection_files[p.parent.name + "/" + p.stem] = full_file_path

    for root, dirs, files in os.walk(images_base_folder):
        length = len(files)
        for file_idx, dir in enumerate(dirs):
            full_file_path = os.path.join(root, dir)
            p = Path(full_file_path)
            if p.parent.name + "/" + p.stem in track_files:
                image_folders[p.parent.name + "/" + p.stem] = full_file_path

    for key, value in track_files.items():
        sequence_tracks = defaultdict(list)
        with open(value, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                frame = row[0]
                tid = int(row[1])
                # to be compatible with geo-referenced bounding boxes we only care about the track id
                sequence_tracks[frame].append((row_idx, tid))

        detections = []
        with (open(detection_files[key], "r", encoding="utf-8") as source):
            for idx, line in enumerate(source):
                if idx == 0:
                    continue
                parts = line.split(" ")
                frame = parts[0]
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                confidence = float(parts[5])
                class_id = int(parts[6])
                detections.append((frame, x1, y1, x2, y2, confidence, class_id))

        colors = set()
        for frame_id in sequence_tracks.keys():
            # Z:\sequences\test\10_1\img1
            image_path = os.path.join(image_folders[key], "img1", frame_id + ".jpg")
            current_sub_folder = deviating_folders(images_base_folder, image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            bounding_boxes = sequence_tracks[frame_id]

            for bb_info in bounding_boxes:
                bb = detections[bb_info[0]]
                min_x = bb[1]
                min_y = bb[2]
                max_x = bb[3]
                max_y = bb[4]
                color = id_to_color(bb_info[1])
                colors.add(color)
                cv2.rectangle(
                    image,
                    (int(min_x), int(min_y)),
                    (int(max_x), int(max_y)),
                    color,
                    thickness=2
                )

            image_target_folder = os.path.join(target_base_folder, Path(current_sub_folder))
            os.makedirs(image_target_folder, exist_ok=True)
            cv2.imwrite(os.path.join(image_target_folder, frame_id + ".jpg"), image)


