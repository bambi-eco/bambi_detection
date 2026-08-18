# -*- coding: utf-8 -*-
"""Transect-based population estimation (naive / bootstrap / ZINB), on arrays.

Port of the R analysis used in Praschl et al. 2026: every transect
contributes one row with a *count* (animals assigned to it) and an *area* in
hectares (the ground it monitored), and three estimators turn that table into
a density in animals per 100 ha (= per km^2):

``naive``
    ``sum(count) / sum(ha) * 100``.
``bootstrap``
    Transects resampled with replacement ``n_boot`` times; mean, SE and
    percentile 95 % CI of the resampled naive densities.
``zinb``
    Zero-inflated negative binomial regression ``count ~ ha`` with a
    constant zero-inflation term (NB2), the R script's
    ``glmmTMB(count ~ ha, ziformula = ~1, family = nbinom2)``, fitted by
    maximum likelihood from a deterministic grid of starts (the likelihood
    is multimodal). Density ``sum(fitted) / sum(ha) * 100`` with fitted
    values on the response scale, SE by the delta method combined as
    ``sqrt(sum(se^2)) / sum(ha) * 100`` - exactly as in R.

The two inputs a survey has to build first, also here:

* **counts** - :func:`assign_to_transects` puts every animal (track) onto
  the transect whose centre line is nearest in perpendicular distance, only
  among transects whose monitored area actually contains it, optionally
  truncated at a maximum distance.
* **areas** - :func:`merged_footprint_area` unions the per-frame ground
  footprints of the frames inside a transect.

Numbers agree with the QGIS plugin's ``core/population.py`` (and through it
with glmmTMB, to 0.02 % on the reference cells) - the plugin's tests are
carried over.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["HA_PER_KM2", "assign_to_transects", "Assignment", "points_in_geometries", "merged_footprint_area",
           "geometry_rings", "NaiveEstimate", "BootstrapEstimate", "ZinbEstimate", "PopulationEstimate",
           "estimate_naive", "estimate_bootstrap", "estimate_zinb", "estimate_population"]

#: Densities are reported per 100 ha, which is exactly 1 km^2.
HA_PER_KM2 = 100.0

# Bounds of the two ZINB parameters whose MLE can run off to infinity - the
# edge of the model, not tuning knobs (see _boundary_params).
LOG_THETA_BOUND = 20.0
GAMMA_BOUND = 20.0


# ---------------------------------------------------------------------------
# Assignment and monitored area
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assignment:
    """Which transect each point was counted into, and why not."""
    transect: NDArray[np.int64]           # (N,) index into the transects, -1 = not counted
    distance: NDArray[np.float64]         # (N,) perpendicular distance to the assigned (or nearest candidate) transect
    nearest_distance: NDArray[np.float64]  # (N,) to the nearest of ALL transects (diagnostics)
    truncated: NDArray[np.bool_]          # (N,) had a transect but was farther than the truncation
    outside: NDArray[np.bool_]            # (N,) no monitored area contains the point
    distances: NDArray[np.float64]        # (N, K) to every transect's centre line

    def counts(self, n_transects: int) -> NDArray[np.int64]:
        """Animals per transect -> ``(K,)``."""
        return np.bincount(self.transect[self.transect >= 0], minlength=int(n_transects)).astype(np.int64)


def assign_to_transects(points: ArrayLike, centerlines: Sequence[ArrayLike], truncation: float = 0.0,
                        inside: Optional[ArrayLike] = None) -> Assignment:
    """Assign every point to a transect it was actually monitored by.

    :param points: ``(N, 2)`` ground positions
    :param centerlines: ``K`` polylines ``(M_k, 2)`` (:func:`bambi.survey.transects.centerline`)
    :param truncation: maximum perpendicular distance in metres; farther points
        stay unassigned. ``0`` (or negative) disables it.
    :param inside: ``(N, K)`` bool - whether transect ``k``'s monitored area
        contains point ``n`` (:func:`points_in_geometries`); ``None`` skips
        the containment test. Without it an animal seen well outside a
        transect's footprint would still be counted into it - whichever
        centre line happened to be nearest - inflating the numerator over
        ground that was never surveyed.
    """
    from bambi.survey.perpendicular import nearest_on_polyline

    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    n, k = len(p), len(centerlines)
    dist = np.full((n, k), np.inf)
    for j, line in enumerate(centerlines):
        line = np.asarray(line, dtype=np.float64)
        if len(line) == 0:
            continue
        dist[:, j] = nearest_on_polyline(p, line)[1]
    finite = np.isfinite(dist)
    if inside is None:
        cand = finite
        outside = np.zeros(n, dtype=bool)
    else:
        ins = np.asarray(inside, dtype=bool).reshape(n, k)
        cand = finite & ins
        outside = ~cand.any(axis=1)
    masked = np.where(cand, dist, np.inf)
    best = np.argmin(masked, axis=1) if k else np.zeros(n, dtype=np.int64)
    has = cand.any(axis=1) if k else np.zeros(n, dtype=bool)
    best_dist = np.where(has, masked[np.arange(n), best] if k else np.inf, np.nan)
    truncating = bool(truncation and truncation > 0)
    truncated = has & (best_dist > truncation) if truncating else np.zeros(n, dtype=bool)
    transect = np.where(has & ~truncated, best, -1).astype(np.int64)
    nearest = np.where(finite.any(axis=1), np.min(np.where(finite, dist, np.inf), axis=1), np.nan) if k \
        else np.full(n, np.nan)
    return Assignment(transect, best_dist, nearest, truncated, outside, dist)


def points_in_geometries(points: ArrayLike, geometries: Sequence) -> NDArray[np.bool_]:
    """``(N, K)`` - whether shapely geometry ``k`` intersects point ``n``
    (a point on the boundary counts as inside). ``None`` geometries contain nothing."""
    from shapely.geometry import Point

    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    out = np.zeros((len(p), len(geometries)), dtype=bool)
    pts = [Point(x, y) for x, y in p]
    for j, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        out[:, j] = [geom.intersects(pt) for pt in pts]
    return out


def merged_footprint_area(footprints: Sequence[ArrayLike]) -> Tuple[float, object]:
    """Union of ground footprints -> ``(area_m2, shapely geometry or None)``.

    Footprints with fewer than three finite corners are skipped;
    self-intersecting ones are repaired with a zero-width buffer (what QGIS'
    ``makeValid`` does).
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    geoms = []
    for fp in footprints:
        q = np.atleast_2d(np.asarray(fp, dtype=np.float64))[:, :2]
        q = q[np.isfinite(q).all(axis=1)]
        if len(q) < 3:
            continue
        poly = Polygon(q)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid and not poly.is_empty:
            geoms.append(poly)
    if not geoms:
        return 0.0, None
    merged = unary_union(geoms)
    if merged.is_empty:
        return 0.0, None
    return float(merged.area), merged


def geometry_rings(geometry) -> List[NDArray[np.float64]]:
    """Exterior rings of a shapely (Multi)Polygon as ``(M, 2)`` arrays (holes dropped - display only)."""
    if geometry is None or geometry.is_empty:
        return []
    polys = getattr(geometry, "geoms", None) or [geometry]
    return [np.asarray(p.exterior.coords, dtype=np.float64)[:, :2] for p in polys if getattr(p, "exterior", None)]


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NaiveEstimate:
    method: str = "naive"
    density_per_100ha: Optional[float] = None
    total_count: float = 0.0
    total_ha: float = 0.0
    abundance_study_area: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapEstimate:
    method: str = "bootstrap"
    density_per_100ha: Optional[float] = None
    se: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None
    n_boot: int = 0
    n_valid: int = 0
    seed: int = 0
    abundance_study_area: Optional[float] = None
    abundance_ci95: Optional[Tuple[float, float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ZinbEstimate:
    method: str = "zinb"
    density_per_100ha: Optional[float] = None
    se: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None
    intercept: Optional[float] = None
    slope_ha: Optional[float] = None
    theta: Optional[float] = None
    zero_inflation_prob: Optional[float] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    converged: bool = False
    dispersion_at_boundary: bool = False
    zero_inflation_at_boundary: bool = False
    abundance_study_area: Optional[float] = None
    abundance_ci95: Optional[Tuple[float, float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["params"] = {"intercept": self.intercept, "ha": self.slope_ha, "theta": self.theta,
                       "zero_inflation_prob": self.zero_inflation_prob}
        return d


@dataclass(frozen=True)
class PopulationEstimate:
    n_transects: int
    total_count: float
    total_ha: float
    n_zero_transects: int
    study_area_ha: float
    naive: Optional[NaiveEstimate] = None
    bootstrap: Optional[BootstrapEstimate] = None
    zinb: Optional[ZinbEstimate] = None

    def to_dict(self) -> Dict[str, object]:
        est = {k: getattr(self, k).to_dict() for k in ("naive", "bootstrap", "zinb") if getattr(self, k) is not None}
        return {"n_transects": self.n_transects, "total_count": self.total_count, "total_ha": self.total_ha,
                "n_zero_transects": self.n_zero_transects, "study_area_ha": self.study_area_ha, "estimates": est}


def estimate_naive(counts: ArrayLike, areas_ha: ArrayLike) -> NaiveEstimate:
    """``sum(count) / sum(ha) * 100`` (animals per 100 ha)."""
    y = np.asarray(counts, dtype=np.float64).ravel()
    ha = np.asarray(areas_ha, dtype=np.float64).ravel()
    total_ha = float(ha.sum())
    if total_ha <= 0:
        return NaiveEstimate(error="total transect area is zero")
    return NaiveEstimate(density_per_100ha=float(y.sum()) / total_ha * HA_PER_KM2, total_count=float(y.sum()),
                         total_ha=total_ha)


def estimate_bootstrap(counts: ArrayLike, areas_ha: ArrayLike, n_boot: int = 999, seed: int = 42
                       ) -> BootstrapEstimate:
    """Bootstrap the naive density by resampling transects with replacement."""
    y = np.asarray(counts, dtype=np.float64).ravel()
    ha = np.asarray(areas_ha, dtype=np.float64).ravel()
    if y.size < 2 or float(ha.sum()) <= 0:
        return BootstrapEstimate(error="need at least 2 transects with a positive area")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y.size, size=(int(n_boot), y.size))
    boot_counts = y[idx].sum(axis=1)
    boot_ha = ha[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(boot_ha > 0, boot_counts / boot_ha * HA_PER_KM2, np.nan)
    valid = dens[np.isfinite(dens)]
    if valid.size == 0:
        return BootstrapEstimate(error="all bootstrap resamples had zero area")
    return BootstrapEstimate(density_per_100ha=float(valid.mean()),
                             se=float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
                             ci95=(float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))),
                             n_boot=int(n_boot), n_valid=int(valid.size), seed=int(seed))


def _zinb_nll(params, y, x):
    """Negative log-likelihood of the ZINB model (NB2, constant inflation)."""
    from scipy.special import gammaln

    b0, b1, log_theta, gamma = params
    eta = np.clip(b0 + b1 * x, -30.0, 30.0)
    mu = np.exp(eta)
    theta = np.exp(np.clip(log_theta, -LOG_THETA_BOUND, LOG_THETA_BOUND))
    gamma = float(np.clip(gamma, -GAMMA_BOUND, GAMMA_BOUND))
    log_p = -np.logaddexp(0.0, -gamma)        # log(sigmoid(gamma))
    log_1mp = -np.logaddexp(0.0, gamma)       # log(1 - sigmoid(gamma))
    log_binom = gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
    log_theta_term = theta * (np.log(theta) - np.log(theta + mu))
    log_mu_term = y * (np.log(mu) - np.log(theta + mu))
    log_nb = log_binom + log_theta_term + log_mu_term
    is_zero = y <= 0
    ll = np.empty_like(mu)
    ll[is_zero] = np.logaddexp(log_p, log_1mp + log_nb[is_zero])
    ll[~is_zero] = log_1mp + log_nb[~is_zero]
    total = float(np.sum(ll))
    return np.inf if not np.isfinite(total) else -total


def _numeric_hessian(fn, params, eps: float = 1e-5):
    p = np.asarray(params, dtype=np.float64)
    n = p.size
    hess = np.zeros((n, n))
    steps = eps * np.maximum(np.abs(p), 1.0)
    for i in range(n):
        for j in range(i, n):
            pi, pj = np.zeros(n), np.zeros(n)
            pi[i] = steps[i]
            pj[j] = steps[j]
            value = (fn(p + pi + pj) - fn(p + pi - pj) - fn(p - pi + pj) + fn(p - pi - pj)) / (4.0 * steps[i] * steps[j])
            hess[i, j] = hess[j, i] = value
    return hess


def _zinb_starts(y, x):
    """Deterministic grid of starting points: the zero-inflation logit is what
    separates the likelihood's modes, so the grid spans "no inflation" through
    "most zeros are structural", crossed with a flat and a data-driven slope."""
    mean_count = float(y.mean())
    positive = y[y > 0]
    conditional_mean = float(positive.mean()) if positive.size else mean_count
    zero_frac = float(np.mean(y <= 0))
    slope = 0.0
    if float(np.ptp(x)) > 0:
        slope = float(np.polyfit(x, np.log1p(y), 1)[0])
    b0_options = [math.log(max(mean_count, 0.1)), math.log(max(conditional_mean, 0.1))]
    b1_options = [0.0, slope] if abs(slope) > 1e-9 else [0.0]
    log_theta_options = [0.0, 1.5]
    gamma_options = [-4.0]
    for share in (0.5, 0.9):
        p = min(max(share * zero_frac, 1e-3), 0.95)
        gamma_options.append(math.log(p / (1.0 - p)))
    return [np.array([b0, b1, lt, g]) for b0 in b0_options for b1 in b1_options
            for lt in log_theta_options for g in gamma_options]


def _boundary_params(log_theta: float, gamma: float):
    fixed, warnings = [], []
    if abs(log_theta) >= LOG_THETA_BOUND - 1e-6:
        fixed.append(2)
        warnings.append("the negative-binomial dispersion is unidentified (the counts reach the Poisson "
                        "limit) - it was held fixed, so the confidence interval understates the true uncertainty")
    if abs(gamma) >= GAMMA_BOUND - 1e-6:
        fixed.append(3)
        warnings.append("zero-inflation collapsed to the boundary - the counts show no excess zeros, so the "
                        "fit reduces to a plain negative binomial")
    return fixed, warnings


def _delta_method_se(nll, params, grad, total_ha, density, fixed):
    n_params = len(params)
    fixed = set(fixed or ())
    free = [i for i in range(n_params) if i not in fixed]
    for i in fixed:
        if np.any(np.abs(grad[:, i]) > 1e-8):
            return None, None, "a boundary parameter still moves the fitted values - no standard errors"
    if not free:
        return None, None, "no free parameters left - no standard errors"
    base = np.asarray(params, dtype=np.float64)

    def profile_nll(sub):
        full = base.copy()
        full[free] = sub
        return nll(full)

    hess = _numeric_hessian(profile_nll, base[free])
    if not np.all(np.isfinite(hess)):
        return None, None, "information matrix is not finite - no standard errors"
    singular = "information matrix is singular - no standard errors"
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return None, None, singular
    if not np.all(np.isfinite(cov)):
        return None, None, singular
    sub_grad = grad[:, free]
    var_fitted = np.einsum("ij,jk,ik->i", sub_grad, cov, sub_grad)
    if not np.all(np.isfinite(var_fitted)) or np.any(var_fitted < 0):
        return None, None, "standard errors are not finite (singular information)"
    se = float(math.sqrt(float(np.sum(var_fitted))) / total_ha * HA_PER_KM2)
    return se, (density - 1.96 * se, density + 1.96 * se), None


def estimate_zinb(counts: ArrayLike, areas_ha: ArrayLike) -> ZinbEstimate:
    """Zero-inflated negative binomial regression of ``count ~ ha``.

    Mirrors ``glmmTMB(count ~ ha, ziformula = ~1, family = nbinom2)`` followed
    by ``augment(type.predict = "response")``. Every start of a deterministic
    grid is optimised (Nelder-Mead then BFGS) and the best likelihood wins -
    that multistart is what reproduces glmmTMB on the multimodal cells.
    """
    from scipy.optimize import minimize

    y = np.asarray(counts, dtype=np.float64).ravel()
    x = np.asarray(areas_ha, dtype=np.float64).ravel()
    total_ha = float(x.sum())
    if y.size < 4 or total_ha <= 0:
        return ZinbEstimate(error="need at least 4 transects with a positive area")
    if np.allclose(x, x[0]):
        return ZinbEstimate(error="all transects have the same area - 'ha' has no variance")

    def nll(params):
        return _zinb_nll(params, y, x)

    best = None
    for start in _zinb_starts(y, x):
        result = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 10000, "xatol": 1e-9, "fatol": 1e-9})
        polished = minimize(nll, result.x, method="BFGS", options={"maxiter": 5000})
        for cand in (result, polished):
            if not np.all(np.isfinite(cand.x)) or not np.isfinite(cand.fun):
                continue
            if best is None or cand.fun < best.fun:
                best = cand
    if best is None:
        return ZinbEstimate(error="model did not converge")
    params = np.clip(best.x, [-np.inf, -np.inf, -LOG_THETA_BOUND, -GAMMA_BOUND],
                     [np.inf, np.inf, LOG_THETA_BOUND, GAMMA_BOUND])
    b0, b1, log_theta, gamma = (float(v) for v in params)
    mu = np.exp(np.clip(b0 + b1 * x, -30.0, 30.0))
    p_zero = float(1.0 / (1.0 + np.exp(-gamma)))
    fitted = (1.0 - p_zero) * mu
    density = float(fitted.sum()) / total_ha * HA_PER_KM2
    grad = np.column_stack([fitted, fitted * x, np.zeros_like(fitted), -fitted * p_zero])
    fixed, warnings = _boundary_params(log_theta, gamma)
    se, ci, se_error = _delta_method_se(nll, params, grad, total_ha, density, fixed)
    if se_error:
        warnings.append(se_error)
    return ZinbEstimate(density_per_100ha=density, se=se, ci95=ci, intercept=b0, slope_ha=b1,
                        theta=float(np.exp(log_theta)), zero_inflation_prob=p_zero,
                        log_likelihood=float(-best.fun), aic=float(2 * 4 + 2 * best.fun),
                        converged=bool(best.success), dispersion_at_boundary=2 in fixed,
                        zero_inflation_at_boundary=3 in fixed, error="; ".join(warnings) if warnings else None)


def _with_study_area(est, study_area_ha: float):
    from dataclasses import replace

    if est is None or est.density_per_100ha is None or not study_area_ha or study_area_ha <= 0:
        return est
    scale = study_area_ha / HA_PER_KM2
    kw = {"abundance_study_area": est.density_per_100ha * scale}
    if getattr(est, "ci95", None):
        kw["abundance_ci95"] = (est.ci95[0] * scale, est.ci95[1] * scale)
    return replace(est, **kw)


def estimate_population(counts: ArrayLike, areas_ha: ArrayLike, methods: Sequence[str] = ("naive", "bootstrap", "zinb"),
                        n_boot: int = 999, seed: int = 42, study_area_ha: float = 0.0) -> PopulationEstimate:
    """Run the requested estimators over the per-transect count/area table.

    :param study_area_ha: when > 0, every density is extrapolated to an
        abundance for a study area of this size (``density / 100 * ha``)
    """
    y = np.asarray(counts, dtype=np.float64).ravel()
    ha = np.asarray(areas_ha, dtype=np.float64).ravel()
    if y.shape != ha.shape:
        raise ValueError("counts and areas_ha must have the same length")
    naive = _with_study_area(estimate_naive(y, ha), study_area_ha) if "naive" in methods else None
    boot = _with_study_area(estimate_bootstrap(y, ha, n_boot, seed), study_area_ha) if "bootstrap" in methods else None
    zinb = _with_study_area(estimate_zinb(y, ha), study_area_ha) if "zinb" in methods else None
    return PopulationEstimate(n_transects=int(y.size), total_count=float(y.sum()), total_ha=float(ha.sum()),
                              n_zero_transects=int((y == 0).sum()), study_area_ha=float(study_area_ha or 0.0),
                              naive=naive, bootstrap=boot, zinb=zinb)
