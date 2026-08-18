# -*- coding: utf-8 -*-
"""Conventional line-transect distance sampling, on arrays.

Given the perpendicular distances of the animals seen from the flight line
and the total length flown, fit a detection function ``g(x)`` - the
probability of seeing an animal at distance ``x`` - by maximum likelihood,
pick the better model by AIC, and turn it into an effective strip width, a
density and an abundance for the covered strip, each with a lognormal 95 %
confidence interval (Buckland et al.). Two detection functions are offered:

* half-normal ``g(x) = exp(-x^2 / 2 sigma^2)``
* hazard-rate ``g(x) = 1 - exp(-(x / sigma)^-b)``

The observed-distance likelihood is ``f(x) = g(x) / mu`` with
``mu = int_0^w g``, the effective strip half-width (ESW). This is the QGIS
plugin's "Distance Sampling" step with files and folders removed; numbers
agree with it to floating point.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["DetectionFunction", "DistanceSamplingResult", "MODELS", "fit_detection_function",
           "estimate_density", "lognormal_ci", "truncation_distance"]

MODELS = ("half-normal", "hazard-rate")


class DetectionFunction(NamedTuple):
    """One fitted detection function."""
    name: str
    params: Tuple[float, ...]        # half-normal: (sigma,); hazard-rate: (sigma, b)
    log_likelihood: float
    aic: float
    esw: float                       # effective strip half-width, metres
    cv_esw: float                    # delta-method CV of the ESW
    g: Callable[[ArrayLike], NDArray[np.float64]]

    @property
    def n_params(self) -> int:
        return len(self.params)


class DistanceSamplingResult(NamedTuple):
    n: int                              # observations after truncation
    n_before_truncation: int
    transect_length_m: float
    truncation_m: float
    best: DetectionFunction
    models: Tuple[DetectionFunction, ...]
    effective_strip_width_m: float
    detection_probability: float
    density_per_km2: float
    density_ci95: Tuple[float, float]
    cv_density: float
    covered_area_km2: float
    abundance_in_covered_area: float
    abundance_ci95: Tuple[float, float]
    curve_x: NDArray[np.float64]        # (60,) distances for plotting g
    curve_g: NDArray[np.float64]
    histogram_counts: NDArray[np.int64]
    histogram_edges: NDArray[np.float64]


def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


def _make_g(name: str, theta: Sequence[float]) -> Callable[[ArrayLike], NDArray[np.float64]]:
    if name == "half-normal":
        sigma = math.exp(theta[0])
        two_var = 2.0 * sigma * sigma
        return lambda t: np.exp(-(np.asarray(t, dtype=np.float64) ** 2) / two_var)
    sigma = math.exp(theta[0])
    b = 1.0 + math.exp(theta[1])

    def g(t):
        t = np.asarray(t, dtype=np.float64)
        out = np.ones_like(t)
        nz = t > 0
        out[nz] = 1.0 - np.exp(-((t[nz] / sigma) ** (-b)))
        return out
    return g


def _esw_of(g, w: float) -> float:
    xs = np.linspace(0.0, w, 512)
    return _trapz(g(xs), xs)


def _params_of(name: str, theta) -> Tuple[float, ...]:
    if name == "half-normal":
        return (math.exp(theta[0]),)
    return (math.exp(theta[0]), 1.0 + math.exp(theta[1]))


def fit_detection_function(distances: ArrayLike, truncation: float, model: str = "half-normal"
                           ) -> Optional[DetectionFunction]:
    """Fit one detection function by MLE to distances in ``[0, w]``.

    :param distances: ``(n,)`` perpendicular distances, metres, already truncated
    :param truncation: ``w``, the truncation distance
    :param model: ``"half-normal"`` or ``"hazard-rate"``
    :return: the fit, or ``None`` when the optimiser did not produce a usable one
    """
    from scipy.optimize import minimize

    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    x = np.asarray(distances, dtype=np.float64).ravel()
    w = float(truncation)
    n = x.size
    if n < 2 or w <= 0:
        return None
    sx2 = float(np.sum(x * x))
    std0 = math.log(max(float(np.std(x)), 1.0))

    if model == "half-normal":
        def nll(theta):
            sigma = math.exp(theta[0])
            mu = _esw_of(_make_g(model, theta), w)
            if mu <= 0:
                return 1e12
            return float(sx2 / (2.0 * sigma * sigma) + n * math.log(mu))
        theta0 = [std0]
    else:
        def nll(theta):
            g = _make_g(model, theta)
            mu = _esw_of(g, w)
            if mu <= 0:
                return 1e12
            gx = np.clip(g(x), 1e-12, 1.0)
            return float(-np.sum(np.log(gx)) + n * math.log(mu))
        theta0 = [std0, 0.0]

    res = minimize(nll, theta0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    theta = np.asarray(res.x, dtype=np.float64)
    best_nll = float(res.fun)
    g = _make_g(model, theta)
    esw = _esw_of(g, w)
    if not math.isfinite(best_nll) or esw <= 0:
        return None
    cv = _cv_esw(nll, lambda th: _esw_of(_make_g(model, th), w), theta)
    k = len(theta0)
    return DetectionFunction(model, _params_of(model, theta), -best_nll, 2.0 * k + 2.0 * best_nll, esw, cv, g)


def _cv_esw(nll, esw_of_theta, theta: NDArray) -> float:
    """Delta-method CV of the ESW from a numeric Hessian of the NLL."""
    m = theta.size
    h = 1e-4 * (np.abs(theta) + 1.0)

    def at(i, di, j, dj):
        t = theta.copy()
        t[i] += di
        t[j] += dj
        return nll(t.tolist())

    hess = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            numer = at(i, h[i], j, h[j]) - at(i, h[i], j, -h[j]) - at(i, -h[i], j, h[j]) + at(i, -h[i], j, -h[j])
            hess[i, j] = hess[j, i] = numer / (4.0 * h[i] * h[j])
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.all(np.isfinite(cov)):
        return 0.0
    grad = np.zeros(m)
    esw0 = esw_of_theta(theta.tolist())
    for i in range(m):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h[i]
        tm[i] -= h[i]
        grad[i] = (esw_of_theta(tp.tolist()) - esw_of_theta(tm.tolist())) / (2.0 * h[i])
    var = float(grad @ cov @ grad)
    if var <= 0 or not math.isfinite(var) or esw0 <= 0:
        return 0.0
    return math.sqrt(var) / esw0


def lognormal_ci(estimate: float, cv: float, z: float = 1.96) -> Tuple[float, float]:
    """95 % lognormal confidence interval of a positive estimate with CV ``cv``."""
    if estimate <= 0 or cv <= 0 or not math.isfinite(cv):
        return (float(estimate), float(estimate))
    c = math.exp(z * math.sqrt(math.log(1.0 + cv * cv)))
    return (estimate / c, estimate * c)


def truncation_distance(distances: ArrayLike, truncation: Optional[float] = None) -> float:
    """``truncation`` when given and > 0, else the 95th percentile of the distances."""
    if truncation is not None and float(truncation) > 0:
        return float(truncation)
    return float(np.percentile(np.asarray(distances, dtype=np.float64), 95))


def estimate_density(distances: ArrayLike, transect_length: float, truncation: Optional[float] = None,
                     models: Sequence[str] = MODELS) -> DistanceSamplingResult:
    """The whole distance-sampling estimate from perpendicular distances.

    :param distances: ``(N,)`` perpendicular distances in metres (non-finite and
        negative values are dropped)
    :param transect_length: total length flown, metres (the effort ``L``)
    :param truncation: ``w`` in metres; ``None``/0 = 95th percentile
    :param models: which detection functions to try; the best AIC wins
    """
    d = np.asarray(distances, dtype=np.float64).ravel()
    d = d[np.isfinite(d) & (d >= 0)]
    if d.size < 2:
        raise ValueError("Not enough perpendicular distances for distance sampling.")
    L = float(transect_length)
    if L <= 0:
        raise ValueError("The transect length must be positive.")
    w = truncation_distance(d, truncation)
    x = d[d <= w]
    n = int(x.size)
    if n < 2:
        raise ValueError("Truncation distance leaves too few observations.")
    fits = [f for f in (fit_detection_function(x, w, m) for m in models) if f is not None]
    if not fits:
        raise ValueError("Detection-function fitting failed for all models.")
    best = min(fits, key=lambda f: f.aic)
    esw = best.esw
    p = esw / w
    density_m2 = n / (2.0 * esw * L)
    density_km2 = density_m2 * 1e6
    covered_m2 = 2.0 * w * L
    abundance = density_m2 * covered_m2
    cv_n = 1.0 / math.sqrt(n)
    cv_density = math.sqrt(cv_n * cv_n + best.cv_esw * best.cv_esw)
    xs = np.linspace(0, w, 60)
    counts, edges = np.histogram(x, bins=min(20, max(5, n // 5)), range=(0, w))
    return DistanceSamplingResult(
        n=n, n_before_truncation=int(d.size), transect_length_m=L, truncation_m=w, best=best,
        models=tuple(fits), effective_strip_width_m=esw, detection_probability=p,
        density_per_km2=density_km2, density_ci95=lognormal_ci(density_km2, cv_density),
        cv_density=cv_density, covered_area_km2=covered_m2 / 1e6, abundance_in_covered_area=abundance,
        abundance_ci95=lognormal_ci(abundance, cv_density), curve_x=xs, curve_g=best.g(xs),
        histogram_counts=counts.astype(np.int64), histogram_edges=edges)
