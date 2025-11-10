import colorsys
import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import cv2

from src.bambi.georeference_deepsort_mot import deviating_folders

# combine to video
# Get-ChildItem *.jpg | Sort-Object Name | ForEach-Object { "file '$($_.Name)'" } | Set-Content list.txt
# ffmpeg -f concat -safe 0 -i list.txt -framerate 30 -c:v libx264 -pix_fmt yuv420p output.mp4


def id_to_color(identifier, *, saturation=0.65, lightness=0.5):
    """
    Deterministically map any identifier (int/str/etc.) to a distinct BGR color for OpenCV.
    """
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2**32  # [0,1)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)  # RGB in [0,1]
    # OpenCV uses BGR and expects ints
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)

    # Bottom edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

    # Left edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)

    # Right edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

if __name__ == '__main__':
    tracks_base_folder = r"Z:\dets\georeferenced_tracks"
    detections_base_folder = r"Z:\dets\source"
    images_base_folder = r"Z:\sequences"
    target_base_folder = r"Z:\sequences_drawn"

    os.makedirs(target_base_folder, exist_ok=True)

    track_files = {}
    detection_files = {}
    image_folders = {}

    # Collect track files (remove the previous one-sequence filter)
    for root, dirs, files in os.walk(tracks_base_folder):
        for file in files:
            if file.endswith(".csv") and "_" in file:
                full_file_path = os.path.join(root, file)
                p = Path(full_file_path)
                track_files[p.parent.name + "/" + p.stem] = full_file_path

    # Match detection files for the same keys
    for root, dirs, files in os.walk(detections_base_folder):
        for file in files:
            full_file_path = os.path.join(root, file)
            p = Path(full_file_path)
            key = p.parent.name + "/" + p.stem
            if key in track_files:
                detection_files[key] = full_file_path

    # Match image folders for the same keys
    for root, dirs, files in os.walk(images_base_folder):
        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            p = Path(full_dir_path)
            key = p.parent.name + "/" + p.stem
            if key in track_files:
                image_folders[key] = full_dir_path

    for key, tracks_csv in track_files.items():
        # 1) Read track-to-frame mapping (frame -> list of (row_idx_in_dets, track_id))
        sequence_tracks = defaultdict(list)
        with open(tracks_csv, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                frame = row[0]
                tid = int(row[1])
                sequence_tracks[frame].append((row_idx, tid))

        # 2) Read detections into a list (the track rows refer to indices here)
        det_path = detection_files.get(key)
        if det_path is None:
            # No detections -> just copy all images without drawing
            dets = []
        else:
            dets = []
            with open(det_path, "r", encoding="utf-8") as source:
                for idx, line in enumerate(source):
                    if idx == 0:
                        continue  # skip header if present
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    frame = parts[0]
                    x1 = float(parts[1])
                    y1 = float(parts[2])
                    x2 = float(parts[3])
                    y2 = float(parts[4])
                    confidence = float(parts[5])
                    class_id = int(parts[6])
                    dets.append((frame, x1, y1, x2, y2, confidence, class_id))

        # 3) Iterate **all images** in the sequence folder
        img_root = image_folders[key]
        img_dir = os.path.join(img_root, "img1")
        if not os.path.isdir(img_dir):
            continue

        # Build/ensure target folder that mirrors the source structure
        # We'll compute the subfolder per image (handles deviating folders)
        for img_file in sorted(Path(img_dir).glob("*.jpg")):
            frame_id = img_file.stem

            image_path = str(img_file)
            current_sub_folder = deviating_folders(images_base_folder, image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            # Draw boxes only if we have tracks for this frame
            if frame_id in sequence_tracks and dets:
                for row_idx, track_id in sequence_tracks[frame_id]:
                    if 0 <= row_idx < len(dets):
                        _, x1, y1, x2, y2, confidence, _ = dets[row_idx]
                        color = id_to_color(track_id)

                        if confidence < 0:
                            cv2.rectangle(
                                image,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color,
                                thickness=2
                            )
                        else:
                            draw_dashed_rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, 15)

            image_target_folder = os.path.join(target_base_folder, Path(current_sub_folder))
            os.makedirs(image_target_folder, exist_ok=True)
            cv2.imwrite(os.path.join(image_target_folder, frame_id + ".jpg"), image)
