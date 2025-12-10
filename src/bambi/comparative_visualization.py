#!/usr/bin/env python3
"""
MOT Tracking Comparison Visualization Tool

Creates grid visualizations comparing ground truth and multiple tracker outputs
for Multi-Object Tracking evaluation.

Usage:
    python visualize_mot_comparison.py <tracking_results_base> <sequences_base> <output_base>

Arguments:
    tracking_results_base: Base folder containing tracker subfolders (e.g., modelbotsort_use_embs1_...)
    sequences_base: Base folder containing test/train subfolders with video sequences
    output_base: Target folder for output visualization grids
"""

import os
import re
import argparse
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.bambi.video.video_writer import FFMPEGWriter


# ============================================================
# STATIC DEFAULT VALUES - Edit these when running without args
# ============================================================
DEFAULT_TRACKING_RESULTS_BASE = r"Z:\ablation2"
DEFAULT_SEQUENCES_BASE = r"Z:\sequences"
DEFAULT_OUTPUT_BASE = r"Z:\ablation_visualization"
DEFAULT_MAX_COLS = 3
DEFAULT_SCALE = 1.0
DEFAULT_SEQUENCES = None # ["143_1"]  # Set to list like ["0_1", "0_2"] to filter, or None for all
DEFAULT_CREATE_VIDEO = True  # Set to True to create videos
DEFAULT_DELETE_IMAGES_AFTER_VIDEO = True  # Set to True to delete images after video creation
DEFAULT_OVERWRITE = False  # Set to True to overwrite existing videos
# ============================================================

def generate_color_palette(num_colors: int = 256) -> Dict[int, Tuple[int, int, int]]:
    """
    Generate a consistent color palette for track IDs.
    Uses HSV color space for perceptually distinct colors.
    """
    colors = {}
    np.random.seed(42)  # For reproducibility

    for i in range(num_colors):
        # Use golden ratio to spread colors evenly in hue space
        hue = int((i * 0.618033988749895 * 180) % 180)
        saturation = 200 + np.random.randint(0, 56)
        value = 200 + np.random.randint(0, 56)

        # Convert HSV to BGR
        hsv = np.uint8([[[hue, saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors[i] = tuple(int(c) for c in bgr[0, 0])

    return colors


def get_color_for_id(track_id: int, color_palette: Dict[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Get a consistent color for a given track ID."""
    return color_palette[track_id % len(color_palette)]


def parse_tracker_folder_name(folder_name: str) -> Optional[Dict[str, any]]:
    """
    Parse tracker configuration from folder name.

    Expected format examples:
        modelbotsort_use_embs1_use_velocity0_use_vectors1
        _use_embs0_use_velocity1_use_vectors0_geo_referenced1
    """
    pattern = re.compile(
        r'(?:model(?P<model_name>[^_]+))?'  # botsort
        r'_use_embs(?P<use_embs>\d+)'  # 0 or 1 etc.
        r'_use_velocity(?P<use_velocity>\d+)'  # 0 or 1 etc.
        r'_use_vectors(?P<use_vectors>\d+)'  # 0 or 1 etc.
        r'(?:_geo_referenced(?P<geo_referenced>\d+))?'  # optional
    )

    match = pattern.search(folder_name)
    if match:
        return {
            'model': match.group('model_name') or 'unknown',
            'use_embs': int(match.group('use_embs')),
            'use_velocity': int(match.group('use_velocity')),
            'use_vectors': int(match.group('use_vectors')),
            'geo_referenced': int(match.group('geo_referenced')) if match.group('geo_referenced') else 0,
            'folder_name': folder_name
        }
    return None


def create_tracker_label(config: Dict) -> str:
    """Create a short label for the tracker configuration."""
    parts = []
    if config['model'] != 'unknown':
        parts.append(config['model'])

    flags = []
    # if config['use_embs']:
    #     flags.append('E')
    # if config['use_velocity']:
    #     flags.append('V')
    # if config['use_vectors']:
    #     flags.append('Vec')
    # if config['geo_referenced']:
    #     flags.append('Geo')

    if flags:
        parts.append('+'.join(flags))

    return ' '.join(parts) if parts else config['folder_name'][:20]


def parse_ground_truth(gt_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    Parse ground truth file.

    Format: frame_id (8 digits), track_id, x, y, w, h, ...
    Returns: {frame_id: [(track_id, x, y, w, h), ...]}
    """
    detections = defaultdict(list)

    if not gt_path.exists():
        return detections

    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 6:
                continue

            try:
                # Ground truth has leading zeros in frame_id
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                detections[frame_id].append((track_id, x, y, w, h))
            except (ValueError, IndexError):
                continue

    return detections


def parse_tracker_results(tracker_path: Path) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    """
    Parse tracker results file.

    Format: frame_id, track_id, x, y, w, h, active, something, confidence
    Returns: {frame_id: [(track_id, x, y, w, h, confidence), ...]}
    """
    detections = defaultdict(list)

    if not tracker_path.exists():
        return detections

    with open(tracker_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 6:
                continue

            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                confidence = float(parts[8]) if len(parts) > 8 else 1.0

                detections[frame_id].append((track_id, x, y, w, h, confidence))
            except (ValueError, IndexError):
                continue

    return detections


def draw_bboxes_on_image(
        image: np.ndarray,
        detections: List[Tuple],
        color_palette: Dict[int, Tuple[int, int, int]],
        label: str,
        is_ground_truth: bool = False,
        show_confidence: bool = False
) -> np.ndarray:
    """
    Draw bounding boxes on image with track IDs.

    Args:
        image: Input image (will be copied)
        detections: List of (track_id, x, y, w, h, [confidence]) tuples
        color_palette: Color mapping for track IDs
        label: Label to show in corner
        is_ground_truth: Whether this is ground truth (affects styling)
        show_confidence: Whether to show confidence scores
    """
    img = image.copy()
    height, width = img.shape[:2]

    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] + 10), (0, 0, 0), -1)
    cv2.putText(img, label, (5, label_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw bounding boxes
    for det in detections:
        track_id = int(det[0])
        x, y, w, h = det[1], det[2], det[3], det[4]
        confidence = det[5] if len(det) > 5 else None

        # Convert to integers
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Get color for this track ID
        color = get_color_for_id(track_id, color_palette)

        # Draw bbox
        thickness = 3 if is_ground_truth else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw track ID label
        id_text = f"ID:{track_id}"
        if show_confidence and confidence is not None:
            id_text += f" ({confidence:.2f})"

        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Background for text
        cv2.rectangle(img, (x1, y1 - text_size[1] - 6), (x1 + text_size[0] + 4, y1), color, -1)

        # Text color (black or white depending on background brightness)
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        cv2.putText(img, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img


def create_grid_image(
        images: List[np.ndarray],
        labels: List[str],
        max_cols: int = 4,
        margin: int = 5
) -> np.ndarray:
    """
    Create a grid of images with labels.

    Args:
        images: List of images to arrange in grid
        labels: Labels for each image (unused here, labels are drawn on images)
        max_cols: Maximum number of columns in grid
        margin: Margin in pixels between images (default: 5)
    """
    if not images:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255

    n = len(images)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    # Get dimensions from first image
    h, w = images[0].shape[:2]

    # Calculate total grid size including margins
    total_width = cols * w + (cols - 1) * margin
    total_height = rows * h + (rows - 1) * margin

    # Ensure dimensions are divisible by 2 (required by FFMPEG/libx264)
    total_width = total_width + (total_width % 2)
    total_height = total_height + (total_height % 2)

    # Create output grid with white background
    grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Resize if necessary
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        # Calculate position with margins
        y_start = row * (h + margin)
        x_start = col * (w + margin)

        grid[y_start:y_start + h, x_start:x_start + w] = img

    return grid


def find_sequences(sequences_base: Path) -> List[Tuple[str, Path]]:
    """
    Find all video sequences in the sequences base folder.

    Returns: List of (sequence_name, sequence_path) tuples
    """
    sequences = []

    for split in ['test', 'train']:
        split_path = sequences_base / split
        if not split_path.exists():
            continue

        for seq_dir in sorted(split_path.iterdir()):
            if seq_dir.is_dir():
                img_dir = seq_dir / 'img1'
                gt_file = seq_dir / 'gt' / 'gt.txt'

                if img_dir.exists():
                    sequences.append((seq_dir.name, seq_dir))

    return sequences


def find_tracker_folders(tracking_base: Path) -> List[Tuple[Dict, Path]]:
    """
    Find all tracker result folders.

    Returns: List of (config_dict, folder_path) tuples
    """
    trackers = []

    for folder in sorted(tracking_base.iterdir()):
        if folder.is_dir():
            config = parse_tracker_folder_name(folder.name)
            if config:
                trackers.append((config, folder))

    return trackers


def get_frame_files(img_dir: Path) -> Dict[int, Path]:
    """
    Get mapping of frame IDs to image file paths.

    Returns: {frame_id: file_path}
    """
    frames = {}

    for img_file in img_dir.glob('*.jpg'):
        try:
            frame_id = int(img_file.stem)
            frames[frame_id] = img_file
        except ValueError:
            continue

    # Also check for png files
    for img_file in img_dir.glob('*.png'):
        try:
            frame_id = int(img_file.stem)
            frames[frame_id] = img_file
        except ValueError:
            continue

    return frames


def process_sequence(
        seq_name: str,
        seq_path: Path,
        trackers: List[Tuple[Dict, Path]],
        output_dir: Path,
        color_palette: Dict[int, Tuple[int, int, int]],
        max_cols: int = 4,
        scale_factor: float = 1.0,
        videowriter: Optional[FFMPEGWriter] = None,
        delete_images_after_video: bool = False,
        overwrite: bool = False
) -> None:
    """
    Process a single video sequence and create comparison grids.

    Args:
        seq_name: Name of the sequence
        seq_path: Path to the sequence folder
        trackers: List of (config, path) tuples for each tracker
        output_dir: Output directory for grids
        color_palette: Color mapping for track IDs
        max_cols: Maximum columns in grid
        scale_factor: Scale factor for images
        videowriter: FFMPEGWriter instance for video creation
        delete_images_after_video: Whether to delete images after video creation
        overwrite: Whether to overwrite existing videos (default: False)
    """
    # Check if video already exists and skip if not overwriting
    video_path = output_dir / f"{seq_name}.mp4"
    if videowriter is not None and video_path.exists() and not overwrite:
        print(f"  Skipping {seq_name} - video already exists (use --overwrite to recreate)")
        return

    img_dir = seq_path / 'img1'
    gt_path = seq_path / 'gt' / 'gt.txt'

    # Get frame files
    frame_files = get_frame_files(img_dir)
    if not frame_files:
        print(f"  No frames found in {img_dir}")
        return

    # Parse ground truth
    gt_detections = parse_ground_truth(gt_path)

    # Parse tracker results for this sequence
    tracker_data = []
    for config, tracker_path in trackers:
        result_file = tracker_path / f"{seq_name}.txt"
        if result_file.exists():
            detections = parse_tracker_results(result_file)
            label = create_tracker_label(config)
            tracker_data.append((label, detections))

    if not tracker_data:
        print(f"  No tracker results found for sequence {seq_name}")
        return

    # Create output directory for this sequence
    seq_output_dir = output_dir / seq_name
    seq_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    frame_ids = sorted(frame_files.keys())
    output_image_paths = []

    for frame_id in tqdm(frame_ids, desc=f"  {seq_name}", leave=False):
        img_path = frame_files[frame_id]

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Scale image if needed
        if scale_factor != 1.0:
            new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
            img = cv2.resize(img, new_size)

        grid_images = []
        grid_labels = []

        # First: Ground truth
        gt_dets = gt_detections.get(frame_id, [])
        if scale_factor != 1.0:
            gt_dets = [(d[0], d[1] * scale_factor, d[2] * scale_factor, d[3] * scale_factor, d[4] * scale_factor) for d
                       in gt_dets]

        gt_img = draw_bboxes_on_image(img, gt_dets, color_palette, "Ground Truth", is_ground_truth=True)
        grid_images.append(gt_img)
        grid_labels.append("Ground Truth")

        # Then: Each tracker
        for label, detections in tracker_data:
            tracker_dets = detections.get(frame_id, [])
            if scale_factor != 1.0:
                tracker_dets = [(d[0], d[1] * scale_factor, d[2] * scale_factor, d[3] * scale_factor,
                                 d[4] * scale_factor, d[5] if len(d) > 5 else 1.0) for d in tracker_dets]

            tracker_img = draw_bboxes_on_image(img, tracker_dets, color_palette, label, show_confidence=True)
            grid_images.append(tracker_img)
            grid_labels.append(label)

        # Create grid
        grid = create_grid_image(grid_images, grid_labels, max_cols=max_cols)

        # Save grid image
        output_path = seq_output_dir / f"{frame_id:08d}.jpg"
        cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        output_image_paths.append(output_path)

    # Create video if requested
    if videowriter is not None and output_image_paths:
        video_path = output_dir / f"{seq_name}.mp4"

        # Sort image files to ensure correct order
        target_image_files = sorted(output_image_paths)

        gen = ((idx, cv2.imread(str(x))) for (idx, x) in enumerate(target_image_files))
        videowriter.write(str(video_path), gen)
        print(f"  Created video {video_path}")

        # Delete images if requested
        if delete_images_after_video:
            for target_image_file in target_image_files:
                os.remove(target_image_file)
            # Also remove the sequence directory if it's empty
            try:
                seq_output_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other error
            print(f"  Deleted {len(target_image_files)} image files")


def main():
    import sys

    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        # Use argument parser
        parser = argparse.ArgumentParser(
            description='Create MOT tracking comparison visualizations',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    python visualize_mot_comparison.py ./tracking_results ./sequences ./output
    python visualize_mot_comparison.py ./tracking_results ./sequences ./output --max-cols 3 --scale 0.5
    python visualize_mot_comparison.py ./tracking_results ./sequences ./output --video --delete-images
            """
        )

        parser.add_argument(
            'tracking_results_base',
            type=str,
            help='Base folder containing tracker result subfolders'
        )
        parser.add_argument(
            'sequences_base',
            type=str,
            help='Base folder containing test/train subfolders with video sequences'
        )
        parser.add_argument(
            'output_base',
            type=str,
            help='Target folder for output visualization grids'
        )
        parser.add_argument(
            '--max-cols',
            type=int,
            default=DEFAULT_MAX_COLS,
            help=f'Maximum number of columns in the grid (default: {DEFAULT_MAX_COLS})'
        )
        parser.add_argument(
            '--scale',
            type=float,
            default=DEFAULT_SCALE,
            help=f'Scale factor for images (default: {DEFAULT_SCALE}, use < 1 for smaller output)'
        )
        parser.add_argument(
            '--sequences',
            type=str,
            nargs='*',
            help='Specific sequence names to process (default: all)'
        )
        parser.add_argument(
            '--video',
            action='store_true',
            default=DEFAULT_CREATE_VIDEO,
            help='Create video from frames using FFMPEGWriter'
        )
        parser.add_argument(
            '--delete-images',
            action='store_true',
            default=DEFAULT_DELETE_IMAGES_AFTER_VIDEO,
            help='Delete image files after video creation (only effective with --video)'
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            default=DEFAULT_OVERWRITE,
            help='Overwrite existing videos (default: skip if video exists)'
        )

        args = parser.parse_args()

        tracking_base = Path(args.tracking_results_base)
        sequences_base = Path(args.sequences_base)
        output_base = Path(args.output_base)
        max_cols = args.max_cols
        scale = args.scale
        filter_sequences = args.sequences
        create_video = args.video
        delete_images_after_video = args.delete_images
        overwrite = args.overwrite
    else:
        # Use static default values
        print("No command line arguments provided, using static default values.")
        print(f"  Tracking results: {DEFAULT_TRACKING_RESULTS_BASE}")
        print(f"  Sequences base:   {DEFAULT_SEQUENCES_BASE}")
        print(f"  Output base:      {DEFAULT_OUTPUT_BASE}")
        print(f"  Max columns:      {DEFAULT_MAX_COLS}")
        print(f"  Scale factor:     {DEFAULT_SCALE}")
        print(f"  Filter sequences: {DEFAULT_SEQUENCES}")
        print(f"  Create video:     {DEFAULT_CREATE_VIDEO}")
        print(f"  Delete images:    {DEFAULT_DELETE_IMAGES_AFTER_VIDEO}")
        print(f"  Overwrite:        {DEFAULT_OVERWRITE}")
        print()

        tracking_base = Path(DEFAULT_TRACKING_RESULTS_BASE)
        sequences_base = Path(DEFAULT_SEQUENCES_BASE)
        output_base = Path(DEFAULT_OUTPUT_BASE)
        max_cols = DEFAULT_MAX_COLS
        scale = DEFAULT_SCALE
        filter_sequences = DEFAULT_SEQUENCES
        create_video = DEFAULT_CREATE_VIDEO
        delete_images_after_video = DEFAULT_DELETE_IMAGES_AFTER_VIDEO
        overwrite = DEFAULT_OVERWRITE

    # Check FFMPEGWriter availability if video creation is requested
    if create_video and FFMPEGWriter is None:
        print("Warning: FFMPEGWriter not available. Install it or adjust the import.")
        print("         Video creation will be skipped.")

    # Validate paths
    if not tracking_base.exists():
        print(f"Error: Tracking results folder not found: {tracking_base}")
        return 1

    if not sequences_base.exists():
        print(f"Error: Sequences folder not found: {sequences_base}")
        return 1

    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)

    # Generate color palette
    print("Generating color palette...")
    color_palette = generate_color_palette(256)

    # Find tracker folders
    print("Finding tracker folders...")
    trackers = find_tracker_folders(tracking_base)
    print(f"  Found {len(trackers)} tracker configurations:")
    for config, path in trackers:
        print(f"    - {create_tracker_label(config)}")

    if not trackers:
        print("Error: No valid tracker folders found")
        return 1

    # Find sequences
    print("Finding video sequences...")
    sequences = find_sequences(sequences_base)
    print(f"  Found {len(sequences)} sequences")

    if not sequences:
        print("Error: No video sequences found")
        return 1

    # Filter sequences if specified
    if filter_sequences:
        sequences = [(name, path) for name, path in sequences if name in filter_sequences]
        print(f"  Filtered to {len(sequences)} sequences")

    videowriter = None
    if create_video:
        videowriter = FFMPEGWriter()

    # Process each sequence
    print("\nProcessing sequences...")
    for seq_name, seq_path in sequences:
        print(f"\nSequence: {seq_name}")
        process_sequence(
            seq_name,
            seq_path,
            trackers,
            output_base,
            color_palette,
            max_cols=max_cols,
            scale_factor=scale,
            videowriter=videowriter,
            delete_images_after_video=delete_images_after_video,
            overwrite=overwrite
        )

    print(f"\nDone! Output saved to: {output_base}")
    return 0


if __name__ == '__main__':
    exit(main())