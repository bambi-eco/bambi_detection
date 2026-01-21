#!/usr/bin/env python3
"""
Track Movement Analyzer

Analyzes geo-referenced tracks to determine if they have moved or remained stable.
A track is considered "moved" if a sufficient number of its bounding box centers
deviate beyond a threshold distance from the track's overall center.

Input format (space or comma separated):
<frame-idx> <track-id> <utm-x-min> <utm-y-min> <utm-z-min> <utm-x-max> <utm-y-max> <utm-z-max> <conf> <class> [<visibility>]
"""

import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class Detection:
    """Single detection/bounding box in a track."""
    frame_idx: int
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float
    confidence: float
    class_id: int
    visibility: float = 1.0

    @property
    def center_2d(self) -> Tuple[float, float]:
        """Return 2D center (x, y) of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )

    @property
    def center_3d(self) -> Tuple[float, float, float]:
        """Return 3D center (x, y, z) of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )


@dataclass
class TrackAnalysis:
    """Analysis results for a single track."""
    track_id: int
    num_detections: int
    track_center: Tuple[float, float, float]
    distances: List[float]
    outlier_indices: List[int]
    outlier_frames: List[int]
    num_outliers: int
    outlier_ratio: float
    is_moved: bool
    max_distance: float
    mean_distance: float
    std_distance: float


def parse_line(line: str) -> Tuple[int, Detection]:
    """
    Parse a single line of the input file.

    Returns:
        Tuple of (track_id, Detection)
    """
    # Handle both comma and space separated formats
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None

    # Try comma-separated first, then space-separated
    if ',' in line:
        parts = [p.strip() for p in line.split(',')]
    else:
        parts = line.split()

    if len(parts) < 10:
        return None, None

    try:
        frame_idx = int(parts[0])
        track_id = int(parts[1])
        x_min = float(parts[2])
        y_min = float(parts[3])
        z_min = float(parts[4])
        x_max = float(parts[5])
        y_max = float(parts[6])
        z_max = float(parts[7])
        confidence = float(parts[8])
        class_id = int(parts[9])
        visibility = float(parts[10]) if len(parts) > 10 else 1.0

        detection = Detection(
            frame_idx=frame_idx,
            x_min=x_min, y_min=y_min, z_min=z_min,
            x_max=x_max, y_max=y_max, z_max=z_max,
            confidence=confidence,
            class_id=class_id,
            visibility=visibility
        )
        return track_id, detection
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
        return None, None


def load_tracks(filepath: Path) -> Dict[int, List[Detection]]:
    """Load all tracks from a file."""
    tracks = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            track_id, detection = parse_line(line)
            if track_id is not None:
                tracks[track_id].append(detection)

    # Sort detections by frame index
    for track_id in tracks:
        tracks[track_id].sort(key=lambda d: d.frame_idx)

    return dict(tracks)


def calculate_track_center(detections: List[Detection], use_3d: bool = False) -> Tuple[float, ...]:
    """Calculate the mean center position of a track."""
    if use_3d:
        centers = np.array([d.center_3d for d in detections])
    else:
        centers = np.array([d.center_2d for d in detections])

    return tuple(np.mean(centers, axis=0))


def calculate_distances(
        detections: List[Detection],
        track_center: Tuple[float, ...],
        use_3d: bool = False
) -> List[float]:
    """Calculate distance of each detection's center from the track center."""
    distances = []
    track_center = np.array(track_center)

    for det in detections:
        if use_3d:
            det_center = np.array(det.center_3d)
        else:
            det_center = np.array(det.center_2d)

        distance = np.linalg.norm(det_center - track_center)
        distances.append(distance)

    return distances


def analyze_track(
        track_id: int,
        detections: List[Detection],
        distance_threshold: float,
        min_outliers_absolute: Optional[int],
        min_outliers_relative: Optional[float],
        use_3d: bool = False
) -> TrackAnalysis:
    """
    Analyze a single track for movement.

    Args:
        track_id: The track identifier
        detections: List of detections for this track
        distance_threshold: Distance threshold in meters for outlier detection
        min_outliers_absolute: Minimum number of outliers to classify as "moved"
        min_outliers_relative: Minimum ratio of outliers to classify as "moved"
        use_3d: Whether to use 3D distance (including Z) or 2D (X, Y only)

    Returns:
        TrackAnalysis with results
    """
    # Calculate track center
    track_center = calculate_track_center(detections, use_3d)

    # Calculate distances
    distances = calculate_distances(detections, track_center, use_3d)

    # Find outliers
    outlier_indices = [i for i, d in enumerate(distances) if d > distance_threshold]
    outlier_frames = [detections[i].frame_idx for i in outlier_indices]
    num_outliers = len(outlier_indices)
    outlier_ratio = num_outliers / len(detections) if detections else 0.0

    # Determine if track has moved
    is_moved = False
    if min_outliers_absolute is not None and min_outliers_relative is not None:
        # Both thresholds: must exceed BOTH (AND logic)
        is_moved = (num_outliers >= min_outliers_absolute) and (outlier_ratio >= min_outliers_relative)
    elif min_outliers_absolute is not None:
        is_moved = num_outliers >= min_outliers_absolute
    elif min_outliers_relative is not None:
        is_moved = outlier_ratio >= min_outliers_relative
    else:
        # Default: any outlier means moved
        is_moved = num_outliers > 0

    return TrackAnalysis(
        track_id=track_id,
        num_detections=len(detections),
        track_center=track_center,
        distances=distances,
        outlier_indices=outlier_indices,
        outlier_frames=outlier_frames,
        num_outliers=num_outliers,
        outlier_ratio=outlier_ratio,
        is_moved=is_moved,
        max_distance=max(distances) if distances else 0.0,
        mean_distance=float(np.mean(distances)) if distances else 0.0,
        std_distance=float(np.std(distances)) if distances else 0.0
    )


def analyze_all_tracks(
        tracks: Dict[int, List[Detection]],
        distance_threshold: float,
        min_outliers_absolute: Optional[int],
        min_outliers_relative: Optional[float],
        use_3d: bool = False,
        outlier_logic: str = "or"
) -> Dict[int, TrackAnalysis]:
    """
    Analyze all tracks for movement.

    Args:
        tracks: Dictionary mapping track_id to list of detections
        distance_threshold: Distance threshold in meters
        min_outliers_absolute: Minimum absolute number of outliers
        min_outliers_relative: Minimum relative ratio of outliers (0.0 - 1.0)
        use_3d: Whether to use 3D distances
        outlier_logic: "or" or "and" - how to combine absolute and relative thresholds

    Returns:
        Dictionary mapping track_id to TrackAnalysis
    """
    results = {}

    for track_id, detections in tracks.items():
        if len(detections) < 2:
            # Skip tracks with only 1 detection - can't determine movement
            continue

        analysis = analyze_track(
            track_id=track_id,
            detections=detections,
            distance_threshold=distance_threshold,
            min_outliers_absolute=min_outliers_absolute,
            min_outliers_relative=min_outliers_relative,
            use_3d=use_3d
        )

        # Apply outlier logic if both thresholds are set
        if min_outliers_absolute is not None and min_outliers_relative is not None:
            if outlier_logic == "or":
                analysis.is_moved = (
                        analysis.num_outliers >= min_outliers_absolute or
                        analysis.outlier_ratio >= min_outliers_relative
                )
            else:  # "and"
                analysis.is_moved = (
                        analysis.num_outliers >= min_outliers_absolute and
                        analysis.outlier_ratio >= min_outliers_relative
                )

        results[track_id] = analysis

    return results


def print_results(
        results: Dict[int, TrackAnalysis],
        verbose: bool = False
) -> None:
    """Print analysis results."""
    moved_tracks = [r for r in results.values() if r.is_moved]
    stable_tracks = [r for r in results.values() if not r.is_moved]

    print("\n" + "=" * 70)
    print("TRACK MOVEMENT ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nTotal tracks analyzed: {len(results)}")
    print(f"Moved tracks:          {len(moved_tracks)}")
    print(f"Stable tracks:         {len(stable_tracks)}")

    print("\n" + "-" * 70)
    print("MOVED TRACKS")
    print("-" * 70)
    if moved_tracks:
        for analysis in sorted(moved_tracks, key=lambda x: x.track_id):
            print(f"\nTrack {analysis.track_id}:")
            print(f"  Detections: {analysis.num_detections}")
            print(f"  Outliers: {analysis.num_outliers} ({analysis.outlier_ratio:.1%})")
            print(f"  Max distance: {analysis.max_distance:.3f} m")
            print(f"  Mean distance: {analysis.mean_distance:.3f} m")
            print(f"  Std distance: {analysis.std_distance:.3f} m")
            if verbose:
                print(f"  Center: ({analysis.track_center[0]:.2f}, {analysis.track_center[1]:.2f})")
                print(
                    f"  Outlier frames: {analysis.outlier_frames[:10]}{'...' if len(analysis.outlier_frames) > 10 else ''}")
    else:
        print("  (none)")

    print("\n" + "-" * 70)
    print("STABLE TRACKS")
    print("-" * 70)
    if stable_tracks:
        for analysis in sorted(stable_tracks, key=lambda x: x.track_id):
            print(f"\nTrack {analysis.track_id}:")
            print(f"  Detections: {analysis.num_detections}")
            print(f"  Outliers: {analysis.num_outliers} ({analysis.outlier_ratio:.1%})")
            print(f"  Max distance: {analysis.max_distance:.3f} m")
            print(f"  Mean distance: {analysis.mean_distance:.3f} m")
            if verbose:
                print(f"  Center: ({analysis.track_center[0]:.2f}, {analysis.track_center[1]:.2f})")
    else:
        print("  (none)")

    print("\n" + "=" * 70)


def export_results(
        results: Dict[int, TrackAnalysis],
        output_path: Path,
        format: str = "csv"
) -> None:
    """Export results to a file."""
    if format == "csv":
        with open(output_path, 'w') as f:
            f.write("track_id,num_detections,is_moved,num_outliers,outlier_ratio,")
            f.write("max_distance,mean_distance,std_distance,center_x,center_y,center_z\n")
            for analysis in sorted(results.values(), key=lambda x: x.track_id):
                center = analysis.track_center
                center_z = center[2] if len(center) > 2 else 0.0
                f.write(f"{analysis.track_id},{analysis.num_detections},{analysis.is_moved},")
                f.write(f"{analysis.num_outliers},{analysis.outlier_ratio:.4f},")
                f.write(f"{analysis.max_distance:.4f},{analysis.mean_distance:.4f},{analysis.std_distance:.4f},")
                f.write(f"{center[0]:.4f},{center[1]:.4f},{center_z:.4f}\n")
        print(f"\nResults exported to: {output_path}")

    elif format == "json":
        import json
        output = {}
        for analysis in results.values():
            output[analysis.track_id] = {
                "num_detections": analysis.num_detections,
                "is_moved": analysis.is_moved,
                "status": "moved" if analysis.is_moved else "stable",
                "num_outliers": analysis.num_outliers,
                "outlier_ratio": round(analysis.outlier_ratio, 4),
                "max_distance": round(analysis.max_distance, 4),
                "mean_distance": round(analysis.mean_distance, 4),
                "std_distance": round(analysis.std_distance, 4),
                "center": [round(c, 4) for c in analysis.track_center],
                "outlier_frames": analysis.outlier_frames
            }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze geo-referenced tracks for movement vs stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 0.5m threshold
  python track_movement_analyzer.py tracks.txt -t 0.5

  # Require at least 3 outliers to be considered "moved"
  python track_movement_analyzer.py tracks.txt -t 0.5 --min-outliers 3

  # Require at least 20%% of detections to be outliers
  python track_movement_analyzer.py tracks.txt -t 0.5 --min-outliers-ratio 0.2

  # Combine both thresholds with OR logic (either condition triggers "moved")
  python track_movement_analyzer.py tracks.txt -t 0.5 --min-outliers 2 --min-outliers-ratio 0.1 --logic or

  # Export results to CSV
  python track_movement_analyzer.py tracks.txt -t 0.5 -o results.csv
        """
    )

    parser.add_argument("--input", type=Path, help="Input file with track data",
                        default=r"Z:\Hugo\mot_georeferenced\test\152_gt.txt")
    parser.add_argument("-t", "--threshold", type=float, default=1.0,
                        help="Distance threshold in meters (default: 1.0)")
    parser.add_argument("--min-outliers", type=int, default=None,
                        help="Minimum absolute number of outliers to be classified as 'moved'")
    parser.add_argument("--min-outliers-ratio", type=float, default=0.2,
                        help="Minimum ratio of outliers (0.0-1.0) to be classified as 'moved'")
    parser.add_argument("--logic", choices=["or", "and"], default="or",
                        help="How to combine absolute and relative thresholds (default: or)")
    parser.add_argument("--use-3d", action="store_true",
                        help="Use 3D distances (including Z coordinate)")
    parser.add_argument("-o", "--output", type=Path, default=r"Z:\Hugo\mot_georeferenced\test\152_gt_classified.csv",
                        help="Output file path (CSV or JSON based on extension)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    if args.min_outliers_ratio is not None:
        if not 0.0 <= args.min_outliers_ratio <= 1.0:
            print("Error: --min-outliers-ratio must be between 0.0 and 1.0")
            return 1

    # Load tracks
    print(f"Loading tracks from: {args.input}")
    tracks = load_tracks(args.input)
    print(f"Loaded {len(tracks)} tracks")

    if not tracks:
        print("No tracks found in input file")
        return 1

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Distance threshold: {args.threshold} m")
    print(f"  Min outliers (absolute): {args.min_outliers if args.min_outliers else 'not set'}")
    print(f"  Min outliers (relative): {f'{args.min_outliers_ratio:.1%}' if args.min_outliers_ratio else 'not set'}")
    if args.min_outliers and args.min_outliers_ratio:
        print(f"  Threshold logic: {args.logic.upper()}")
    print(f"  Use 3D distances: {args.use_3d}")

    # Analyze tracks
    results = analyze_all_tracks(
        tracks=tracks,
        distance_threshold=args.threshold,
        min_outliers_absolute=args.min_outliers,
        min_outliers_relative=args.min_outliers_ratio,
        use_3d=args.use_3d,
        outlier_logic=args.logic
    )

    # Print results
    print_results(results, verbose=args.verbose)

    # Export if requested
    if args.output:
        format = "json" if args.output.suffix.lower() == ".json" else "csv"
        export_results(results, args.output, format=format)

    return 0


if __name__ == "__main__":
    exit(main())