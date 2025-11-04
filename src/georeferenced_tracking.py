import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from src.bambi.georeference_deepsort_mot import deviating_folders


@dataclass
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

@dataclass
class Track:
    tid: int
    cls: Optional[int]
    x1: float
    y1: float
    x2: float
    y2: float
    last_frame: int
    age: int = 0      # number of consecutive frames not matched
    hits: int = 0     # number of total matches

def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0

def _center(b):
    x1, y1, x2, y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _diag(b):
    x1, y1, x2, y2 = b
    return math.hypot(x2 - x1, y2 - y1)

def _bbox(d):
    return (d.x1, d.y1, d.x2, d.y2)

def postprocess_merge_fragments(
    results: List[Tuple[int, int, object]],
    min_len: int = 4,           # minimum frames to be considered a valid track
    max_gap: int = 2,           # allow stitching across gaps up to this many frames
    min_iou: float = 0.5,       # IOU threshold for stitching
    max_center_ratio: float = 0.5,  # allow stitching if center distance <= ratio * avg diagonal
    class_aware: bool = True    # only stitch tracks with same class_id
) -> List[Tuple[int, int, object]]:
    """
    Returns a *new* results list with short tracks reassigned to the closest valid track
    when they're temporally adjacent and spatially close.
    """
    # Build per-track timelines
    tracks: Dict[int, List[Tuple[int, object]]] = defaultdict(list)
    for f, tid, d in results:
        tracks[tid].append((f, d))
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])

    # Identify valid vs fragment tracks
    valid_tids = {tid for tid, seq in tracks.items() if len(seq) >= min_len}
    frag_tids  = [tid for tid, seq in tracks.items() if len(seq) < min_len]

    # Precompute quick boundary info for each track
    # (first frame/box/class, last frame/box/class)
    boundary = {}
    for tid, seq in tracks.items():
        f0, d0 = seq[0]
        f1, d1 = seq[-1]
        boundary[tid] = {
            "first_frame": f0, "first_box": _bbox(d0), "first_cls": d0.cls,
            "last_frame":  f1, "last_box":  _bbox(d1), "last_cls":  d1.cls
        }

    # Helper to decide stitching between (A -> B) using boundaries:
    # direction = "forward" means A ends before B starts (A then B).
    def _stitch_score(a_tid: int, b_tid: int, direction: str) -> Optional[dict]:
        a, b = boundary[a_tid], boundary[b_tid]
        if class_aware and a["first_cls"] != b["first_cls"]:
            return None

        if direction == "forward":
            gap = b["first_frame"] - a["last_frame"]
            if gap < 0 or (gap > max_gap and max_gap > 0):
                return None
            box_a = a["last_box"]; box_b = b["first_box"]
        else:  # "backward": A starts after B ends, so B then A
            gap = a["first_frame"] - b["last_frame"]
            if gap < 0 or (gap > max_gap and max_gap > 0):
                return None
            box_a = b["last_box"]; box_b = a["first_box"]

        iou = _bbox_iou(box_a, box_b)
        ca = _center(box_a); cb = _center(box_b)
        center_dist = math.hypot(ca[0]-cb[0], ca[1]-cb[1])
        norm = 0.5*(_diag(box_a) + _diag(box_b))
        norm = max(norm, 1e-6)
        center_ratio = center_dist / norm

        ok = (iou >= min_iou) or (center_ratio <= max_center_ratio)
        if not ok:
            return None
        # Lower score is better; we combine (1-IOU) and center_ratio
        score = (1.0 - iou) + center_ratio
        return {"score": score, "gap": gap, "iou": iou, "center_ratio": center_ratio}

    # Decide merges for each fragment
    # We do not allow frame conflicts: fragment frames must be disjoint from the target.
    def _no_frame_overlap(frag_tid: int, target_tid: int) -> bool:
        frag_frames = {f for f, _ in tracks[frag_tid]}
        target_frames = {f for f, _ in tracks[target_tid]}
        return frag_frames.isdisjoint(target_frames)

    reassignment: Dict[int, int] = {}  # frag_tid -> chosen valid target tid

    for frag_tid in frag_tids:
        # Try forward and backward stitching
        best = None
        best_target = None

        # Candidates: valid tracks only
        for vt in valid_tids:
            if not _no_frame_overlap(frag_tid, vt):
                continue

            # forward: frag ends before valid starts
            s1 = _stitch_score(frag_tid, vt, "forward")
            # backward: valid ends before frag starts
            s2 = _stitch_score(frag_tid, vt, "backward")

            for s in (s1, s2):
                if s is None:
                    continue
                if (best is None) or (s["score"] < best["score"]) or \
                   (math.isclose(s["score"], best["score"]) and s["gap"] < best["gap"]):
                    best = s
                    best_target = vt

        if best_target is not None:
            reassignment[frag_tid] = best_target

    # Apply reassignments (note: if multiple frags map to same valid track, we just append)
    new_results: List[Tuple[int, int, object]] = []
    for f, tid, d in results:
        if tid in reassignment:
            new_tid = reassignment[tid]
            new_results.append((f, new_tid, d))
        else:
            new_results.append((f, tid, d))

    # Optional: you may want to drop now-empty track IDs; at this point they remain,
    # but consumers usually just look at (frame, track_id). To clean up, recompute:
    # (We’ll reindex nothing; just return merged assignments.)
    new_results.sort(key=lambda r: (r[0], r[1]))
    return new_results

def parse_line(line: str) -> Optional[Detection]:
    """
    Expect lines like:
      frame_id min_x, min_y, min_z, max_x, max_y, max_z, confidence, class_id
    Commas or spaces are accepted. z-values are ignored.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # allow both comma and whitespace separated
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) < 9:
        raise ValueError(f"Line has too few fields ({len(parts)}): {line}")

    frame = int(parts[0])
    x1 = float(parts[1]); y1 = float(parts[2])
    # parts[3] is min_z (ignored)
    x2 = float(parts[4]); y2 = float(parts[5])
    # parts[6] is max_z (ignored)
    conf = float(parts[7])
    try:
        cls = int(parts[8])
    except ValueError:
        # fallback because initial export is messed up ^^
        cls = int(float(parts[8]))
    # Ensure x1<=x2 and y1<=y2
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return Detection(frame, x1, y1, x2, y2, conf, cls)

def read_detections(path: str) -> Dict[int, List[Detection]]:
    frames = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            det = parse_line(line)
            if det is not None:
                frames[det.frame].append(det)
    return dict(frames)

def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom

def greedy_match(
    detections: List[Detection],
    tracks: List[Track],
    iou_thr: float,
    class_aware: bool
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy assignment by descending IOU.
    Returns:
      matches: list of (det_idx, track_idx_in_tracks_list)
      unmatched_dets: indices of detections not matched
      unmatched_trks: indices of tracks not matched
    """
    pairs = []
    for di, d in enumerate(detections):
        for ti, t in enumerate(tracks):
            if class_aware and t.cls is not None and d.cls != t.cls:
                continue
            i = iou((d.x1, d.y1, d.x2, d.y2), (t.x1, t.y1, t.x2, t.y2))
            if i >= iou_thr:
                pairs.append((i, di, ti))
    # sort by IOU desc
    pairs.sort(key=lambda x: x[0], reverse=True)

    det_used = set()
    trk_used = set()
    matches = []
    for i, di, ti in pairs:
        if di in det_used or ti in trk_used:
            continue
        det_used.add(di)
        trk_used.add(ti)
        matches.append((di, ti))

    unmatched_dets = [i for i in range(len(detections)) if i not in det_used]
    unmatched_trks = [i for i in range(len(tracks)) if i not in trk_used]
    return matches, unmatched_dets, unmatched_trks

def track_detections(
    frames: Dict[int, List[Detection]],
    iou_thr: float = 0.9,
    class_aware: bool = True,
    max_age: int = 2
) -> List[Tuple[int, int, Detection]]:
    """
    Run IOU-based tracking.

    Returns list of tuples (frame_id, track_id, detection) in chronological order.
    """
    all_frames = sorted(frames.keys())
    active_tracks: List[Track] = []
    next_tid = 1

    results: List[Tuple[int, int, Detection]] = []

    for f in all_frames:
        dets = frames[f]
        # match current detections to active tracks
        matches, unmatched_dets, unmatched_trks = greedy_match(dets, active_tracks, iou_thr, class_aware)

        # update matched tracks
        for di, ti in matches:
            d = dets[di]
            t = active_tracks[ti]
            t.x1, t.y1, t.x2, t.y2 = d.x1, d.y1, d.x2, d.y2
            t.last_frame = f
            t.hits += 1
            t.age = 0
            results.append((f, t.tid, d))

        # create new tracks for unmatched detections
        for di in unmatched_dets:
            d = dets[di]
            t = Track(
                tid=next_tid,
                cls=(d.cls if class_aware else None),
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                last_frame=f,
                age=0, hits=1
            )
            active_tracks.append(t)
            results.append((f, t.tid, d))
            next_tid += 1

        # age unmatched tracks and drop stale ones
        survivors = []
        for ti in range(len(active_tracks)):
            if ti in unmatched_trks:
                active_tracks[ti].age += 1
            if max_age < 0 or active_tracks[ti].age <= max_age:
                survivors.append(active_tracks[ti])
        active_tracks = survivors

    # sort results strictly by (frame, track_id) for stable output
    results.sort(key=lambda r: (r[0], r[1]))
    return results

def write_tracks_csv(results: List[Tuple[int, int, Detection]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        # f.write("frame_id track_id min_x min_y max_x max_y confidence class_id\n")
        for frame, tid, d in results:
            f.write(f"{frame:08d},{tid},{d.x1:.6f},{d.y1:.6f},{d.x2:.6f},{d.y2:.6f},{d.conf:.6f},{d.cls}\n")

if __name__ == '__main__':
    # Paths
    base_dir = r"Z:\dets\georeferenced"
    target_dir = r"Z:\dets\georeferenced_tracks"
    iou_thresh = 0.6
    class_aware = True
    max_age = -1

    ##############################

    os.makedirs(target_dir, exist_ok=True)

    # Loop through both directories
    for root, dirs, files in os.walk(base_dir):
        length = len(files)
        for file_idx, file in enumerate(files):
            if file.endswith(".txt") and "_" in file:
                full_file_path = os.path.join(root, file)
                print(f"{file_idx + 1} / {length}: {full_file_path}")
                frames = read_detections(os.path.join(root, full_file_path))
                results = track_detections(
                    frames,
                    iou_thr=iou_thresh,
                    class_aware=class_aware,
                    max_age=max_age
                )

                results = postprocess_merge_fragments(
                    results,
                    min_len=15,  # fragments are tracks with < x detections
                    max_gap=max_age,  # allow up to y empty frames between pieces
                    min_iou=0,  # accept if boundary IOU ≥ x
                    max_center_ratio=0.75,  # or if centers within x × avg diag
                    class_aware=class_aware
                )

                tracks = defaultdict(list)
                for r in results:
                    tracks[r[1]].append(r)
                num_tracks = len(tracks)
                x = 0
                for t in tracks.values():
                    x += len(t)
                p = Path(file)
                target_folder = os.path.join(target_dir, deviating_folders(base_dir, full_file_path))
                os.makedirs(target_folder, exist_ok=True)
                target_file = os.path.join(target_folder, p.stem + ".csv")
                write_tracks_csv(results, target_file)
                break
            break
