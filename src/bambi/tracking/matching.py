# -*- coding: utf-8 -*-
"""Cross-modal track matching: which thermal track and which RGB track are the
same animal - on arrays.

Implements section 3.2 of *When One Modality Is Not Enough*. Three steps:

1. **Frame correspondence** by capture time (:func:`match_frames_by_time`) -
   the two cameras run at different rates, so "the same frame" only means
   anything on the shared clock.
2. **A 2D affine** ``T: RGB -> thermal`` fitted from corresponding detection
   centres, which absorbs the different fields of view, resolutions and the
   small parallax between the two lenses.
3. **Track-level assignment**: every RGB track is compared to every thermal
   track by the *median* inter-centre distance over the frames they share,
   and the pairing is one-to-one.

Two departures from the paper, both deliberate:

* **The affine is bootstrapped, not given.** The paper fits ``T`` from
  corresponding centres, but on real data the correspondence is exactly what
  is being solved for. We seed from frames where each modality holds exactly
  one detection - where correspondence is not in doubt - and then alternate
  assignment and refitting.
* **Assignment is Hungarian, not greedy.** The paper takes the smallest
  median distance first; a global optimum costs nothing extra here and cannot
  be led astray by an unlucky first pick.

What confirmation *means* is the point: a pair seen in both modalities is a
real animal - of 34 unmatched tracks in the paper only one was real.

Every input is a set of parallel arrays per modality (``frames``,
``track_ids``, ``centres (N, 2)``, ``confidences``) plus the ``(K, 2)``
frame correspondence; results are arrays indexed the same way. Ported from
the QGIS plugin's ``core/track_matching.py`` + ``core/frame_matching.py``,
which were already pure NumPy/SciPy inside but dict-shaped at the edges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Affine", "AffineFit", "Detections", "Candidates", "MatchConfig", "MatchResult",
           "match_frames_by_time", "fit_affine", "affine_rmse", "seed_pairs", "estimate_affine",
           "candidates", "passes_gates", "assign", "rejection_reasons", "match_tracks", "detections"]

#: Cost written into gated-out cells. ``linear_sum_assignment`` needs a finite
#: matrix, so an impossible pair gets a number no real distance can reach and
#: is dropped again after the assignment.
_BLOCKED = 1e9


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MatchConfig(NamedTuple):
    """Gates a candidate pair has to clear to be called the same animal."""

    #: Frames the two tracks must have in common. Below this the median is not
    #: a median of anything.
    min_shared: int = 8
    #: Maximum median inter-centre distance, in thermal pixels. 28 px is the
    #: paper's empirical gate at their 1024x1024 inference size; it is exposed
    #: because it does not transfer to a different resolution unchanged.
    gate_px: float = 28.0
    #: Mean detection confidence both tracks must reach.
    min_confidence: float = 0.20
    #: How far apart two frames may be on the shared clock and still be
    #: called the same moment, in seconds.
    max_time_offset: float = 0.10


@dataclass(frozen=True)
class Detections:
    """One modality's tracked boxes, as centres. Build with :func:`detections`."""
    frames: NDArray[np.int64]
    track_ids: NDArray[np.int64]
    centres: NDArray[np.float64]        # (N, 2)
    confidences: NDArray[np.float64]
    detection_ids: NDArray[np.int64]    # caller's ids, carried through to the pairs

    def __len__(self) -> int:
        return len(self.frames)


def detections(frames: ArrayLike, track_ids: ArrayLike, boxes: ArrayLike,
               confidences: Optional[ArrayLike] = None,
               detection_ids: Optional[ArrayLike] = None) -> Detections:
    """``(N, 4)`` xyxy boxes (or ``(N, 2)`` centres) + ids -> :class:`Detections`."""
    fr = np.asarray(frames).astype(np.int64).ravel()
    tid = np.asarray(track_ids).astype(np.int64).ravel()
    b = np.asarray(boxes, dtype=np.float64)
    n = len(fr)
    if b.ndim != 2 or b.shape[0] != n or b.shape[1] not in (2, 4) or len(tid) != n:
        raise ValueError("frames, track_ids and boxes (N, 4) / centres (N, 2) must have the same length")
    centres = b if b.shape[1] == 2 else 0.5 * (b[:, :2] + b[:, 2:4])
    conf = np.zeros(n) if confidences is None else np.nan_to_num(np.asarray(confidences, dtype=np.float64).ravel())
    ids = np.arange(n, dtype=np.int64) if detection_ids is None else np.asarray(detection_ids).astype(np.int64).ravel()
    if len(conf) != n or len(ids) != n:
        raise ValueError("confidences and detection_ids must be (N,)")
    return Detections(fr, tid, centres, conf, ids)


@dataclass(frozen=True)
class Affine:
    """``q = A p + t``, mapping an RGB image point onto the thermal image."""
    matrix: NDArray[np.float64]         # (2, 2)
    offset: NDArray[np.float64]         # (2,)

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return p @ self.matrix.T + self.offset

    @property
    def coefficients(self) -> Tuple[float, float, float, float, float, float]:
        """``(a, b, c, d, tx, ty)`` - the plugin's spelling."""
        (a, b), (c, d) = self.matrix
        return float(a), float(b), float(c), float(d), float(self.offset[0]), float(self.offset[1])

    def inverse_scale(self) -> Tuple[float, float]:
        """How many RGB pixels one thermal pixel spans, per axis (column norms
        of the linear part, so a rotated transform still gives the true scale)."""
        sx = float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))
        sy = float(np.hypot(self.matrix[0, 1], self.matrix[1, 1]))
        return (1.0 / sx if sx else 1.0, 1.0 / sy if sy else 1.0)

    def to_list(self):
        """``[[[a, b], [c, d]], [tx, ty]]`` - the form the plugin's match store persists."""
        return [self.matrix.tolist(), self.offset.tolist()]

    @classmethod
    def from_list(cls, payload) -> "Affine":
        if not payload:
            return cls.identity()
        return cls(np.asarray(payload[0], dtype=np.float64).reshape(2, 2),
                   np.asarray(payload[1], dtype=np.float64).reshape(2))

    @classmethod
    def from_coefficients(cls, a, b, c, d, tx, ty) -> "Affine":
        return cls(np.array([[a, b], [c, d]], dtype=np.float64), np.array([tx, ty], dtype=np.float64))

    @classmethod
    def identity(cls) -> "Affine":
        return cls(np.eye(2), np.zeros(2))

    @classmethod
    def scaling(cls, sx: float, sy: float) -> "Affine":
        return cls(np.diag([float(sx), float(sy)]), np.zeros(2))

    def __eq__(self, other) -> bool:
        return isinstance(other, Affine) and np.array_equal(self.matrix, other.matrix) \
            and np.array_equal(self.offset, other.offset)


class AffineFit(NamedTuple):
    affine: Affine
    rmse: float
    n_pairs: int
    seeded: bool        # True: started from unambiguous frames; False: from the frame-size scale


# ---------------------------------------------------------------------------
# Frame correspondence
# ---------------------------------------------------------------------------

def match_frames_by_time(src_epochs: ArrayLike, dst_epochs: ArrayLike,
                         max_dt: float = MatchConfig().max_time_offset) -> NDArray[np.int64]:
    """For every source frame the destination frame closest in time, or ``-1``.

    Thermal and RGB are recorded by two cameras on one aircraft at different
    frame rates, so "frame 100" means nothing across them - but both stamp
    every frame with the same real-world clock. Nearest on that clock, within
    ``max_dt`` seconds, is the only correspondence that holds; at the ends of a
    flight the nearest frame can be seconds away, and that is a ``-1``.

    :param src_epochs: ``(N,)`` capture time in epoch seconds (NaN = unknown)
    :param dst_epochs: ``(M,)`` likewise
    :return: ``(N,)`` index into ``dst`` or ``-1``
    """
    src = np.asarray(src_epochs, dtype=np.float64).ravel()
    dst = np.asarray(dst_epochs, dtype=np.float64).ravel()
    out = np.full(len(src), -1, dtype=np.int64)
    valid_dst = np.flatnonzero(np.isfinite(dst))
    if len(valid_dst) == 0:
        return out
    order = valid_dst[np.argsort(dst[valid_dst], kind="stable")]
    keys = dst[order]
    ok = np.isfinite(src)
    pos = np.searchsorted(keys, src[ok])
    lo = np.clip(pos - 1, 0, len(keys) - 1)
    hi = np.clip(pos, 0, len(keys) - 1)
    d_lo = np.abs(keys[lo] - src[ok])
    d_hi = np.abs(keys[hi] - src[ok])
    best = np.where(d_hi < d_lo, hi, lo)              # ties -> the earlier frame, like bisect_left
    d = np.minimum(d_lo, d_hi)
    res = np.where(d <= max_dt, order[best], -1)
    out[ok] = res
    return out


def frame_pairs_from_map(frame_map: ArrayLike) -> NDArray[np.int64]:
    """``(N,)`` per-frame map (``-1`` = none) -> ``(K, 2)`` ``[src_frame, dst_frame]`` pairs."""
    m = np.asarray(frame_map).astype(np.int64).ravel()
    src = np.flatnonzero(m >= 0)
    return np.column_stack([src, m[src]])


# ---------------------------------------------------------------------------
# The affine
# ---------------------------------------------------------------------------

def fit_affine(src: ArrayLike, dst: ArrayLike) -> Optional[Affine]:
    """Least-squares ``T`` with ``dst ~ T(src)`` from ``(N, 2)`` point pairs.

    ``None`` when there is too little to fit, or when the points are
    degenerate (all on one line, or all at one place) - an affine through
    collinear points is unconstrained perpendicular to that line, and would
    send every off-line detection somewhere arbitrary.
    """
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) < 3 or len(s) != len(d):
        return None
    design = np.column_stack([s, np.ones(len(s))])
    if np.linalg.matrix_rank(design, tol=1e-6) < 3:
        return None
    solution, *_ = np.linalg.lstsq(design, d, rcond=None)      # (3, 2): rows = [a c; b d; tx ty]
    if not np.all(np.isfinite(solution)):
        return None
    return Affine(solution[:2, :].T.copy(), solution[2, :].copy())


def affine_rmse(affine: Affine, src: ArrayLike, dst: ArrayLike) -> float:
    """Root-mean-square residual of ``affine`` over the pairs, in pixels."""
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) == 0:
        return float("inf")
    r = affine.apply(s) - d
    return float(np.sqrt(np.mean(np.sum(r * r, axis=1))))


# ---------------------------------------------------------------------------
# Bootstrapping the affine
# ---------------------------------------------------------------------------

def _rows_by_frame(det: Detections) -> Dict[int, NDArray[np.int64]]:
    order = np.argsort(det.frames, kind="stable")
    f = det.frames[order]
    if len(f) == 0:
        return {}
    bounds = np.flatnonzero(np.diff(f)) + 1
    return {int(g[0]): idx for g, idx in zip(np.split(f, bounds), np.split(order, bounds))}


def _pairs(frame_pairs: ArrayLike) -> NDArray[np.int64]:
    fp = np.asarray(frame_pairs).astype(np.int64).reshape(-1, 2)
    return fp[np.argsort(fp[:, 0], kind="stable")]


def seed_pairs(det_t: Detections, det_w: Detections, frame_pairs: ArrayLike
               ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Correspondences from frames where neither modality is ambiguous.

    A frame holding exactly one detection in each modality pairs them without
    any assumption about the transform - which is what makes it possible to
    estimate the transform at all.

    :return: ``(rgb (S, 2), thermal (S, 2))`` centres
    """
    by_t, by_w = _rows_by_frame(det_t), _rows_by_frame(det_w)
    src, dst = [], []
    for ft, fw in _pairs(frame_pairs):
        it, iw = by_t.get(int(ft)), by_w.get(int(fw))
        if it is not None and iw is not None and len(it) == 1 and len(iw) == 1:
            src.append(det_w.centres[iw[0]])
            dst.append(det_t.centres[it[0]])
    if not src:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.array(src), np.array(dst)


def _assign_within_frame(affine: Affine, c_t: NDArray, c_w: NDArray) -> Tuple[NDArray, NDArray]:
    """Pair one frame's detections by nearest mapped centre, one-to-one -> (rows_w, cols_t)."""
    from scipy.optimize import linear_sum_assignment

    if len(c_t) == 0 or len(c_w) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    mapped = affine.apply(c_w)
    cost = np.linalg.norm(mapped[:, None, :] - c_t[None, :, :], axis=2)
    return linear_sum_assignment(cost)


def _size_fallback(frame_size_t, frame_size_w) -> Affine:
    """Scale RGB onto thermal by frame size alone - the last resort."""
    if not frame_size_t or not frame_size_w:
        return Affine.identity()
    wt, ht = frame_size_t
    ww, hw = frame_size_w
    if not ww or not hw:
        return Affine.identity()
    return Affine.scaling(wt / ww, ht / hw)


def _refine(affine: Affine, best_rmse: float, n_pairs: int, by_t, by_w, fp, det_t, det_w,
            max_iterations: int) -> Tuple[Affine, float, int]:
    """Alternate one-to-one assignment within frames and refitting."""
    for _ in range(max_iterations):
        s_list, d_list = [], []
        for ft, fw in fp:
            it, iw = by_t.get(int(ft)), by_w.get(int(fw))
            if it is None or iw is None:
                continue
            rows_w, cols_t = _assign_within_frame(affine, det_t.centres[it], det_w.centres[iw])
            if len(rows_w):
                s_list.append(det_w.centres[iw][rows_w])
                d_list.append(det_t.centres[it][cols_t])
        if not s_list:
            break
        s = np.vstack(s_list)
        d = np.vstack(d_list)
        candidate = fit_affine(s, d)
        if candidate is None:
            break
        rmse = affine_rmse(candidate, s, d)
        # Judge the refit on the SAME pairs the current transform is scored on.
        # (The plugin compared against the previous iteration's residual over
        # a different, smaller pair set - a seed fitted exactly through three
        # points scored 0 and could never be improved upon, however wrong.)
        current = affine_rmse(affine, s, d)
        if not (rmse < current - 1e-6):
            if not math.isfinite(best_rmse) or len(s) > n_pairs:
                best_rmse, n_pairs = current, len(s)      # report the residual over everything paired
            break
        affine, best_rmse, n_pairs = candidate, rmse, len(s)
    return affine, best_rmse, n_pairs


def estimate_affine(det_t: Detections, det_w: Detections, frame_pairs: ArrayLike,
                    frame_size_t: Optional[Tuple[float, float]] = None,
                    frame_size_w: Optional[Tuple[float, float]] = None,
                    max_iterations: int = 5) -> AffineFit:
    """Estimate ``T: RGB -> thermal``.

    Starts from the best initial guess available and then alternates
    one-to-one assignment with refitting until the residual stops improving:

    * **Unambiguous frames** - one detection in each modality - pair without
      assuming anything about the transform, so they are the preferred start.
    * **Failing that, the pure scale implied by the two frame sizes.** A herd
      can put several animals in *every* frame, leaving no unambiguous frame
      at all - the normal case on the paper's flights. The scale guess is good
      enough because both cameras look down from one airframe.

    Both starts are refined when a seed exists, and the one that ends with the
    lower residual wins: a seed built from three noisy points can be exactly
    fitted and still wrong, and the alternation then converges to a wrong
    pairing; the scale start does not share that failure. ``rmse`` says
    whether the result can be trusted; ``inf`` means nothing could be fitted
    and the start guess is returned unchanged.
    """
    src, dst = seed_pairs(det_t, det_w, frame_pairs)
    seed = fit_affine(src, dst)
    by_t, by_w = _rows_by_frame(det_t), _rows_by_frame(det_w)
    fp = _pairs(frame_pairs)

    fallback = _size_fallback(frame_size_t, frame_size_w)
    from_scale = _refine(fallback, float("inf"), 0, by_t, by_w, fp, det_t, det_w, max_iterations)
    if seed is None:
        affine, rmse, n = from_scale
        return AffineFit(affine, rmse, n, False)
    from_seed = _refine(seed, affine_rmse(seed, src, dst), len(src), by_t, by_w, fp, det_t, det_w, max_iterations)
    if math.isfinite(from_scale[1]) and from_scale[1] < from_seed[1] - 1e-6:
        affine, rmse, n = from_scale
        return AffineFit(affine, rmse, n, False)
    affine, rmse, n = from_seed
    return AffineFit(affine, rmse, n, True)


# ---------------------------------------------------------------------------
# Track-level cost
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidates:
    """Every (thermal track, RGB track) pair sharing at least one frame, sorted
    by ``(median_dist, track_id_t, track_id_w)``, plus the per-frame evidence.

    ``pair_*`` arrays hold one row per co-occurrence; ``pair_candidate``
    indexes into the candidate arrays.
    """
    track_id_t: NDArray[np.int64]
    track_id_w: NDArray[np.int64]
    shared: NDArray[np.int64]
    median_dist: NDArray[np.float64]
    conf_t: NDArray[np.float64]
    conf_w: NDArray[np.float64]
    pair_candidate: NDArray[np.int64]
    pair_frame_t: NDArray[np.int64]
    pair_frame_w: NDArray[np.int64]
    pair_detection_t: NDArray[np.int64]
    pair_detection_w: NDArray[np.int64]
    pair_dist: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.track_id_t)

    def pairs_of(self, index: int) -> NDArray[np.int64]:
        """Row indices of the ``pair_*`` arrays that belong to candidate ``index``."""
        return np.flatnonzero(self.pair_candidate == index)


def _mean_confidence(det: Detections) -> Dict[int, float]:
    if len(det) == 0:
        return {}
    ids, inv = np.unique(det.track_ids, return_inverse=True)
    sums = np.bincount(inv, weights=det.confidences)
    counts = np.bincount(inv)
    return {int(t): float(s / c) for t, s, c in zip(ids, sums, counts)}


def candidates(det_t: Detections, det_w: Detections, frame_pairs: ArrayLike, affine: Affine) -> Candidates:
    """Every track pair sharing at least one frame, with its median distance.

    The *median* rather than the mean is what makes this robust: a track that
    is briefly confused with a neighbour, or a frame where one box is badly
    placed, moves a mean enough to break a real pair.
    """
    by_t, by_w = _rows_by_frame(det_t), _rows_by_frame(det_w)
    it_all, iw_all = [], []
    for ft, fw in _pairs(frame_pairs):
        it, iw = by_t.get(int(ft)), by_w.get(int(fw))
        if it is None or iw is None:
            continue
        g_t, g_w = np.meshgrid(it, iw, indexing="ij")           # thermal outer, rgb inner: the plugin's order
        it_all.append(g_t.ravel())
        iw_all.append(g_w.ravel())
    empty_i = np.zeros(0, np.int64)
    empty_f = np.zeros(0, np.float64)
    if not it_all:
        return Candidates(empty_i, empty_i, empty_i, empty_f, empty_f, empty_f,
                          empty_i, empty_i, empty_i, empty_i, empty_i, empty_f)
    it = np.concatenate(it_all)
    iw = np.concatenate(iw_all)
    dist = np.linalg.norm(affine.apply(det_w.centres[iw]) - det_t.centres[it], axis=1)
    key = np.column_stack([det_t.track_ids[it], det_w.track_ids[iw]])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(uniq)
    shared = np.bincount(inv, minlength=n)
    median = np.array([np.median(dist[inv == k]) for k in range(n)])
    conf_t_map, conf_w_map = _mean_confidence(det_t), _mean_confidence(det_w)
    conf_t = np.array([conf_t_map.get(int(t), 0.0) for t in uniq[:, 0]])
    conf_w = np.array([conf_w_map.get(int(w), 0.0) for w in uniq[:, 1]])
    order = np.lexsort((uniq[:, 1], uniq[:, 0], median))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    pair_cand = rank[inv]
    p_order = np.argsort(pair_cand, kind="stable")           # group the evidence per candidate, in-frame order kept
    return Candidates(uniq[order, 0], uniq[order, 1], shared[order], median[order], conf_t[order], conf_w[order],
                      pair_cand[p_order], det_t.frames[it][p_order], det_w.frames[iw][p_order],
                      det_t.detection_ids[it][p_order], det_w.detection_ids[iw][p_order], dist[p_order])


def passes_gates(cands: Candidates, config: MatchConfig = MatchConfig()) -> NDArray[np.bool_]:
    """Whether each candidate is admissible at all, before any assignment."""
    return ((cands.shared >= config.min_shared) & (cands.median_dist < config.gate_px)
            & (cands.conf_t >= config.min_confidence) & (cands.conf_w >= config.min_confidence))


def assign(cands: Candidates, config: MatchConfig = MatchConfig()) -> NDArray[np.int64]:
    """One-to-one assignment over the gated candidates.

    Hungarian rather than the paper's greedy pass: the matrices are at most a
    few dozen square, so the global optimum is free, and greedy can spend a
    track on a near miss that a later, better pair then needs.

    :return: indices into ``cands`` of the confirmed pairs, closest first
    """
    from scipy.optimize import linear_sum_assignment

    ok = np.flatnonzero(passes_gates(cands, config))
    if len(ok) == 0:
        return np.zeros(0, np.int64)
    tracks_t, it = np.unique(cands.track_id_t[ok], return_inverse=True)
    tracks_w, iw = np.unique(cands.track_id_w[ok], return_inverse=True)
    cost = np.full((len(tracks_t), len(tracks_w)), _BLOCKED, dtype=float)
    lookup = np.full((len(tracks_t), len(tracks_w)), -1, dtype=np.int64)
    cost[it, iw] = cands.median_dist[ok]
    lookup[it, iw] = ok
    rows, cols = linear_sum_assignment(cost)
    # A rectangular problem forces every row (or column) to take something,
    # including a blocked cell, so the gate is re-applied after the fact.
    keep = cost[rows, cols] < _BLOCKED
    chosen = lookup[rows[keep], cols[keep]]
    return chosen[np.argsort(cands.median_dist[chosen], kind="stable")]


def rejection_reasons(cands: Candidates, config: MatchConfig = MatchConfig()) -> Dict[str, int]:
    """Why candidates were turned away, for the run log. "No matches" is
    otherwise indistinguishable from "the gate is wrong"."""
    too_few = cands.shared < config.min_shared
    too_far = ~too_few & (cands.median_dist >= config.gate_px)
    low = ~too_few & ~too_far & (np.minimum(cands.conf_t, cands.conf_w) < config.min_confidence)
    return {"shared_frames": int(too_few.sum()), "distance": int(too_far.sum()), "confidence": int(low.sum())}


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

class MatchResult(NamedTuple):
    matches: NDArray[np.int64]      # indices into ``candidates`` of the confirmed pairs, closest first
    candidates: Candidates
    affine: Affine
    affine_rmse: float
    n_candidates: int
    n_rejected: int


def match_tracks(det_t: Detections, det_w: Detections, frame_pairs: ArrayLike,
                 config: MatchConfig = MatchConfig(),
                 frame_size_t: Optional[Tuple[float, float]] = None,
                 frame_size_w: Optional[Tuple[float, float]] = None) -> MatchResult:
    """Match thermal tracks to RGB tracks. The whole of section 3.2 in one call.

    :param det_t: thermal detections (:func:`detections`)
    :param det_w: RGB detections
    :param frame_pairs: ``(K, 2)`` ``[thermal_frame, rgb_frame]`` taken at the
        same moment (:func:`match_frames_by_time` + :func:`frame_pairs_from_map`)
    """
    fit = estimate_affine(det_t, det_w, frame_pairs, frame_size_t, frame_size_w)
    cands = candidates(det_t, det_w, frame_pairs, fit.affine)
    accepted = assign(cands, config)
    return MatchResult(accepted, cands, fit.affine, fit.rmse, len(cands), len(cands) - len(accepted))
