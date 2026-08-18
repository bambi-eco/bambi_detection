# -*- coding: utf-8 -*-
"""bambi.tracking.iou - the array tracker against the scalar reference and by hand."""
import numpy as np
import pytest

from bambi.tracking.iou import Tracks, interpolate_tracks, iou_matrix, split_boxes, track_boxes

scipy = pytest.importorskip("scipy")


# ------------------------------------------------------------------ helpers

def _random_scene(seed, n_frames=40, n_animals=6, drop=0.15, jitter=0.4, size=3.0, classes=2, split_prob=0.02):
    """Animals walking on a plane, boxes jittered, some detections dropped, a
    few extra spurious boxes: enough structure to exercise every branch."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 60, size=(n_animals, 2))
    vel = rng.normal(0, 0.6, size=(n_animals, 2))
    cls = rng.integers(0, classes, size=n_animals)
    frames, boxes, klass, conf = [], [], [], []
    for f in range(n_frames):
        pos = pos + vel
        for a in range(n_animals):
            if rng.random() < drop:
                continue
            c = pos[a] + rng.normal(0, jitter, 2)
            s = size * rng.uniform(0.8, 1.2)
            frames.append(f)
            boxes.append([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2])
            klass.append(cls[a])
            conf.append(rng.uniform(0.3, 1.0))
        if rng.random() < split_prob * 10:                   # a spurious box
            c = rng.uniform(0, 60, 2)
            frames.append(f); boxes.append([c[0], c[1], c[0] + 2, c[1] + 2]); klass.append(rng.integers(0, classes)); conf.append(0.2)
    return np.array(frames), np.array(boxes, float), np.array(klass), np.array(conf)


def _reference_ids(frames, boxes, classes, mode, iou_thr, class_aware, max_age, max_dist):
    """Run bambi.georeferenced_tracking.track_detections and map its ids back."""
    from collections import defaultdict
    from bambi.georeferenced_tracking import Detection, TrackerMode, track_detections

    by_frame = defaultdict(list)
    for i, (f, b, c) in enumerate(zip(frames, boxes, classes)):
        by_frame[int(f)].append(Detection(source_id=i, frame=int(f), x1=b[0], y1=b[1], z1=0.0,
                                          x2=b[2], y2=b[3], z2=0.0, conf=1.0, cls=int(c)))
    results = track_detections(by_frame, iou_thr=iou_thr, class_aware=class_aware, max_age=max_age,
                               tracker_mode=TrackerMode[mode.upper()], max_center_distance=max_dist)
    ids = np.zeros(len(frames), dtype=np.int64)
    for _f, tid, d in results:
        ids[d.source_id] = tid
    return ids


# ------------------------------------------------------------------ iou

def test_iou_matrix_values():
    a = [[0, 0, 2, 2]]
    b = [[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6], [0, 0, 0, 0]]
    m = iou_matrix(a, b)
    assert m.dtype == np.float32 and m.shape == (1, 4)
    assert np.allclose(m[0], [1 / 7, 1.0, 0.0, 0.0])


def test_split_boxes():
    xy, z = split_boxes([[1, 2, 3, 4, 5, 6]])
    assert np.array_equal(xy, [[1, 2, 4, 5]]) and np.array_equal(z, [[3, 6]])
    assert split_boxes([[1, 2, 3, 4]])[1] is None
    with pytest.raises(ValueError):
        split_boxes([[1, 2, 3]])


# ------------------------------------------------------------------ tracker vs reference

@pytest.mark.parametrize("mode", ["greedy", "hungarian", "center", "hungarian_center"])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("class_aware,max_age", [(True, -1), (False, 2), (True, 0)])
def test_matches_the_scalar_reference_row_for_row(mode, seed, class_aware, max_age):
    frames, boxes, classes, _ = _random_scene(seed)
    got = track_boxes(frames, boxes, classes, mode=mode, iou_threshold=0.3, class_aware=class_aware,
                      max_age=max_age, max_center_distance=2.5)
    want = _reference_ids(frames, boxes, classes, mode, 0.3, class_aware, max_age, 2.5)
    assert np.array_equal(got, want)


def test_input_order_within_a_frame_is_respected_even_when_unsorted():
    frames, boxes, classes, _ = _random_scene(5)
    perm = np.random.default_rng(0).permutation(len(frames))
    got = track_boxes(frames[perm], boxes[perm], classes[perm])
    # the reference groups by frame in *input* order; feed it the same permuted order
    want = _reference_ids(frames[perm], boxes[perm], classes[perm], "hungarian", 0.3, True, -1, 10.0)
    assert np.array_equal(got, want)


def test_six_column_boxes_track_on_the_horizontal_extent():
    frames, boxes, classes, _ = _random_scene(7)
    z = np.random.default_rng(1).uniform(0, 5, size=(len(frames), 1))
    six = np.hstack([boxes[:, :2], z, boxes[:, 2:], z + 1])
    assert np.array_equal(track_boxes(frames, six, classes), track_boxes(frames, boxes, classes))


def test_hand_checked_small_case():
    # two animals, frames 0..3; animal B skips frame 2; a third box appears at frame 3
    frames = [0, 0, 1, 1, 2, 3, 3, 3]
    boxes = [[0, 0, 2, 2], [10, 10, 12, 12],
             [0.2, 0, 2.2, 2], [10.1, 10, 12.1, 12],
             [0.4, 0, 2.4, 2],
             [0.6, 0, 2.6, 2], [10.3, 10, 12.3, 12], [50, 50, 52, 52]]
    ids = track_boxes(frames, boxes, mode="hungarian", iou_threshold=0.3)
    assert ids.tolist() == [1, 2, 1, 2, 1, 1, 2, 3]
    # with max_age 0 animal B is closed after frame 1 and gets a new id at frame 3
    ids0 = track_boxes(frames, boxes, mode="hungarian", iou_threshold=0.3, max_age=0)
    assert ids0.tolist() == [1, 2, 1, 2, 1, 1, 3, 4]


def test_class_aware_never_joins_classes():
    frames = [0, 1]
    boxes = [[0, 0, 2, 2], [0, 0, 2, 2]]
    assert track_boxes(frames, boxes, [0, 1]).tolist() == [1, 2]
    assert track_boxes(frames, boxes, [0, 1], class_aware=False).tolist() == [1, 1]
    assert track_boxes(frames, boxes).tolist() == [1, 1]           # no classes at all


def test_empty_and_bad_input():
    assert track_boxes([], np.zeros((0, 4))).shape == (0,)
    with pytest.raises(ValueError):
        track_boxes([0], [[0, 0, 1, 1]], mode="magic")
    with pytest.raises(ValueError):
        track_boxes([0, 1], [[0, 0, 1, 1]])


# ------------------------------------------------------------------ interpolation

def test_interpolation_fills_gaps_and_marks_them():
    frames = [0, 3, 1, 0]
    tids = [1, 1, 2, 2]
    boxes = [[0, 0, 1, 1], [3, 3, 4, 4], [5, 5, 6, 6], [5, 5, 6, 6]]
    conf = [0.2, 0.8, 1.0, 1.0]
    t = interpolate_tracks(frames, tids, boxes, conf, [7, 7, 9, 9])
    assert isinstance(t, Tracks) and len(t) == 6 and t.n_tracks == 2
    assert t.frames.tolist() == [0, 0, 1, 1, 2, 3]
    assert t.track_ids.tolist() == [1, 2, 1, 2, 1, 1]
    assert t.interpolated.tolist() == [False, False, True, False, True, False]
    assert t.source.tolist() == [0, 3, -1, 2, -1, 1]
    assert np.allclose(t.boxes[2], [1, 1, 2, 2]) and np.allclose(t.boxes[4], [2, 2, 3, 3])
    assert np.allclose(t.confidences[[2, 4]], 0.5)                     # mean, the plugin's rule
    lin = interpolate_tracks(frames, tids, boxes, conf, confidence="linear")
    assert np.allclose(lin.confidences[[2, 4]], [0.4, 0.6])
    assert t.classes.tolist() == [7, 9, 7, 9, 7, 7]


def test_interpolation_matches_the_reference_module():
    from bambi.georeferenced_tracking import Detection, interpolate_missing_frames

    frames, boxes, classes, conf = _random_scene(11, drop=0.35)
    ids = track_boxes(frames, boxes, classes)
    ours = interpolate_tracks(frames, ids, boxes, conf, classes, confidence="linear")
    ref_in = [(int(f), int(t), Detection(i, int(f), b[0], b[1], 0.0, b[2], b[3], 0.0, float(c), int(k)))
              for i, (f, t, b, c, k) in enumerate(zip(frames, ids, boxes, conf, classes))]
    ref = interpolate_missing_frames(ref_in)
    assert len(ref) == len(ours)
    for (f, t, d), i in zip(ref, range(len(ours))):
        assert f == ours.frames[i] and t == ours.track_ids[i]
        assert np.allclose([d.x1, d.y1, d.x2, d.y2], ours.boxes[i])
        assert d.conf == pytest.approx(ours.confidences[i]) and d.cls == ours.classes[i]
        assert bool(d.interpolated) == bool(ours.interpolated[i])


def test_no_fill_just_sorts():
    t = interpolate_tracks([5, 1], [1, 1], [[0, 0, 1, 1], [9, 9, 10, 10]], fill=False)
    assert t.frames.tolist() == [1, 5] and t.source.tolist() == [1, 0] and not t.interpolated.any()
