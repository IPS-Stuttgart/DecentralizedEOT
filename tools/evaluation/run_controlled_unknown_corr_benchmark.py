#!/usr/bin/env python3
"""Controlled unknown-correlation benchmark for RED-style ellipse-track fusion.

This script is intentionally separate from the full tracker simulation.  It
creates pairs of local object-level track estimates with a known hidden common
information source, random RED chart ambiguity, and controllable correlation
strength.  It then evaluates fusion rules that either know the common
information, assume independence, or operate under unknown correlation.

The benchmark is useful for CI because it is much faster than the full point
cloud / MEM-EKF* tracking simulation and it exposes the failure modes that are
hard to isolate in the moving-target benchmark:

* RED normal fusion: assumes independence and can become overconfident.
* RED-IF correct common info: reference for known common information.
* RED-IF wrong common info: fragility when common information is mis-specified.
* Parameter-CI: conservative but not RED-aware, so it can fuse the wrong chart.
* RED-CI / RED-ICI: unknown-correlation methods with RED chart alignment.
* RED-CI-ESR / RED-GCI-ESR: geometry-aware variants for the square-root/MMGW
  shape representation.

The state order matches the Göttingen RED-IF repository:
    [x, y, vx, vy, alpha, length, width]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.special import logsumexp
from scipy.stats import chi2

try:  # IoU is useful but should not make the benchmark unusable.
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover - dependency guard for minimal setups
    Polygon = None

X1, X2, V1, V2, AL, L, W = range(7)
STATE_DIM = 7
SHAPE_DIM = 3
EPS = 1e-10

DEFAULT_METHODS = [
    "oracle_correlated_red",
    "red_normal",
    "red_if_correct",
    "red_if_wrong",
    "parameter_ci",
    "red_ci",
    "red_ici",
    "red_ci_esr",
    "red_gci_esr",
]

METHOD_LABELS = {
    "oracle_correlated_red": "Oracle correlated RED fusion",
    "red_normal": "RED normal fusion",
    "red_if_correct": "RED-IF, correct common info",
    "red_if_wrong": "RED-IF, wrong common info",
    "parameter_ci": "Parameter-CI",
    "red_ci": "RED-CI",
    "red_ici": "RED-ICI",
    "red_ci_esr": "RED-CI-ESR",
    "red_gci_esr": "RED-GCI/ESR",
}


@dataclass
class FusionOutput:
    mean: np.ndarray
    covariance: np.ndarray
    valid: bool = True
    omega: float = np.nan


@dataclass
class TrialInputs:
    true_state: np.ndarray
    common_mean: np.ndarray
    common_cov: np.ndarray
    est_a: np.ndarray
    cov_a: np.ndarray
    est_b_observed: np.ndarray
    cov_b_observed: np.ndarray
    cross_ab_observed: np.ndarray
    chart_b: int


@dataclass
class MetricAccumulator:
    gw: List[float]
    iou: List[float]
    nees: List[float]
    logdet: List[float]
    runtime_ms: List[float]
    invalid: int = 0

    def append(self, out: FusionOutput, true_state: np.ndarray, runtime_ms: float) -> None:
        if not out.valid or not np.all(np.isfinite(out.mean)) or not np.all(np.isfinite(out.covariance)):
            self.invalid += 1
            self.gw.append(np.nan)
            self.iou.append(np.nan)
            self.nees.append(np.nan)
            self.logdet.append(np.nan)
            self.runtime_ms.append(runtime_ms)
            return

        cov = ensure_spd(out.covariance)
        self.gw.append(float(gw_error(out.mean, true_state)))
        self.iou.append(float(iou(out.mean, true_state)))
        self.nees.append(float(nees_error(out.mean, cov, true_state)))
        self.logdet.append(float(logdet_spd(cov)))
        self.runtime_ms.append(float(runtime_ms))

    def summary(self) -> Dict[str, float]:
        return {
            "mean_gw": nanmean(self.gw),
            "median_gw": nanmedian(self.gw),
            "mean_iou": nanmean(self.iou),
            "mean_nees": nanmean(self.nees),
            "nees_ratio": nanmean(self.nees) / STATE_DIM,
            "mean_logdet": nanmean(self.logdet),
            "mean_runtime_ms": nanmean(self.runtime_ms),
            "invalid_rate": self.invalid / max(len(self.runtime_ms), 1),
        }


def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else float("nan")


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_method_list(text: str) -> List[str]:
    methods = [x.strip() for x in text.split(",") if x.strip()]
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Known: {DEFAULT_METHODS}")
    return methods


def symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(mat, dtype=float) + np.asarray(mat, dtype=float).T)


def ensure_spd(mat: np.ndarray, jitter: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    mat = symmetrize(mat)
    eye = np.eye(mat.shape[0])
    cur = mat.copy()
    scale = jitter
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(cur)
            return cur
        except np.linalg.LinAlgError:
            cur = symmetrize(cur + scale * eye)
            scale *= 10.0
    vals, vecs = np.linalg.eigh(cur)
    vals = np.maximum(vals, jitter)
    return symmetrize(vecs @ np.diag(vals) @ vecs.T)


def inv_spd(mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(ensure_spd(mat))


def logdet_spd(mat: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(ensure_spd(mat))
    if sign <= 0:
        return float("nan")
    return float(value)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def rot(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def shape_matrix(state: np.ndarray, square_root: bool = False) -> np.ndarray:
    power = 1.0 if square_root else 2.0
    length = max(float(abs(state[L])), 1e-6)
    width = max(float(abs(state[W])), 1e-6)
    return rot(float(state[AL])) @ np.diag([length ** power, width ** power]) @ rot(float(state[AL])).T


def shape_sqrt_params(shape_state: np.ndarray) -> np.ndarray:
    alpha, length, width = np.asarray(shape_state, dtype=float)
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([
        length * c * c + width * s * s,
        (length - width) * s * c,
        length * s * s + width * c * c,
    ])


def shape_sqrt_jacobian(shape_state: np.ndarray) -> np.ndarray:
    alpha, length, width = np.asarray(shape_state, dtype=float)
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([
        [2.0 * s * c * (width - length), c * c, s * s],
        [(length - width) * (c * c - s * s), s * c, -s * c],
        [2.0 * s * c * (length - width), s * s, c * c],
    ])


def sqrt_params_to_shape(sqrt_params: np.ndarray) -> np.ndarray:
    mat = np.array([[sqrt_params[0], sqrt_params[1]], [sqrt_params[1], sqrt_params[2]]], dtype=float)
    mat = ensure_spd(mat[:2, :2])
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    alpha = math.atan2(vecs[1, 0], vecs[0, 0])
    return np.array([alpha, float(vals[0]), float(vals[1])])


def full_esr_transform_jacobian(state: np.ndarray) -> np.ndarray:
    jac = np.zeros((STATE_DIM, STATE_DIM))
    jac[:4, :4] = np.eye(4)
    jac[4:7, 4:7] = shape_sqrt_jacobian(state[[AL, L, W]])
    return jac


def transform_chart_matrix(k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Affine matrix/offset for an equivalent RED chart transform."""
    a = np.eye(STATE_DIM)
    b = np.zeros(STATE_DIM)
    b[AL] = k * 0.5 * np.pi
    if k % 2:
        a[[L, W], :] = a[[W, L], :]
    return a, b


def transform_chart(
    mean: np.ndarray,
    cov: np.ndarray,
    k: int,
    cross_ab: Optional[np.ndarray] = None,
    transform_side: str = "b",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    a, b = transform_chart_matrix(k)
    new_mean = a @ mean + b
    new_cov = ensure_spd(a @ cov @ a.T)
    new_cross = None
    if cross_ab is not None:
        # cross_ab = Cov(A, B).  If we transform B, C_AB -> C_AB A^T.
        # If we transform A, C_AB -> A C_AB.
        if transform_side.lower() == "b":
            new_cross = cross_ab @ a.T
        elif transform_side.lower() == "a":
            new_cross = a @ cross_ab
        else:
            raise ValueError("transform_side must be 'a' or 'b'")
    return new_mean, new_cov, new_cross


def chart_distance(candidate: np.ndarray, reference: np.ndarray) -> float:
    shape_term = (
        angle_diff(candidate[AL], reference[AL]) ** 2
        + (candidate[L] - reference[L]) ** 2
        + (candidate[W] - reference[W]) ** 2
    )
    kin_term = 0.05 * float(np.sum((candidate[:4] - reference[:4]) ** 2))
    return float(shape_term + kin_term)


def align_to_reference(
    mean: np.ndarray,
    cov: np.ndarray,
    reference: np.ndarray,
    cross_ab: Optional[np.ndarray] = None,
    transform_side: str = "b",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    best = None
    for k in range(-4, 5):
        cur_mean, cur_cov, cur_cross = transform_chart(mean, cov, k, cross_ab, transform_side=transform_side)
        score = chart_distance(cur_mean, reference)
        if best is None or score < best[0]:
            best = (score, cur_mean, cur_cov, cur_cross, k)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def canonical_error(mean: np.ndarray, true_state: np.ndarray) -> np.ndarray:
    aligned, _, _, _ = align_to_reference(mean, np.eye(STATE_DIM), true_state)
    err = aligned - true_state
    err[AL] = angle_diff(aligned[AL], true_state[AL])
    return err


def gw_error(est: np.ndarray, true_state: np.ndarray) -> float:
    gt_sigma = symmetrize(shape_matrix(true_state))
    track_sigma = symmetrize(shape_matrix(est))
    gt_sqrt = sqrtm(gt_sigma)
    inner = sqrtm(gt_sqrt @ track_sigma @ gt_sqrt)
    val = (
        np.linalg.norm(true_state[[X1, X2]] - est[[X1, X2]]) ** 2
        + np.trace(gt_sigma + track_sigma - 2.0 * inner)
    )
    return float(np.real_if_close(val))


def ellipse_polygon(state: np.ndarray, n: int = 128):
    if Polygon is None:
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = state[[X1, X2], None] + rot(state[AL]) @ np.diag([abs(state[L]), abs(state[W])]) @ np.vstack([
        np.cos(theta),
        np.sin(theta),
    ])
    return Polygon(pts.T)


def iou(est: np.ndarray, true_state: np.ndarray) -> float:
    if Polygon is None:
        return float("nan")
    p_est = ellipse_polygon(est)
    p_true = ellipse_polygon(true_state)
    if p_est is None or p_true is None or not p_est.is_valid or not p_true.is_valid:
        return float("nan")
    denom = p_est.area + p_true.area - p_est.intersection(p_true).area
    return float(p_est.intersection(p_true).area / denom) if denom > 0 else float("nan")


def nees_error(est: np.ndarray, cov: np.ndarray, true_state: np.ndarray) -> float:
    err = canonical_error(est, true_state)
    return float(err.T @ inv_spd(cov) @ err)


def independent_fusion(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> FusionOutput:
    ya = inv_spd(cov_a)
    yb = inv_spd(cov_b)
    info = ya + yb
    cov = inv_spd(info)
    mean = cov @ (ya @ mean_a + yb @ mean_b)
    return FusionOutput(mean=mean, covariance=cov)


def oracle_correlated_fusion(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    cross_ab: np.ndarray,
) -> FusionOutput:
    cov_joint = ensure_spd(np.block([[cov_a, cross_ab], [cross_ab.T, cov_b]]))
    h = np.vstack([np.eye(STATE_DIM), np.eye(STATE_DIM)])
    z = np.concatenate([mean_a, mean_b])
    inv_joint = inv_spd(cov_joint)
    cov = inv_spd(h.T @ inv_joint @ h)
    mean = cov @ h.T @ inv_joint @ z
    return FusionOutput(mean=mean, covariance=cov)


def red_if_fusion(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    common_mean: np.ndarray,
    common_cov: np.ndarray,
    gamma: float = 1.0,
) -> FusionOutput:
    ya = inv_spd(cov_a)
    yb = inv_spd(cov_b)
    y0 = inv_spd(common_cov)
    info = ya + yb - gamma * y0
    valid = True
    try:
        np.linalg.cholesky(symmetrize(info))
    except np.linalg.LinAlgError:
        valid = False
    cov = inv_spd(info)
    mean = cov @ (ya @ mean_a + yb @ mean_b - gamma * y0 @ common_mean)
    return FusionOutput(mean=mean, covariance=cov, valid=valid)


def covariance_objective(cov: np.ndarray, mean: np.ndarray, criterion: str) -> float:
    cov = ensure_spd(cov)
    if criterion in {"esr_trace", "esr_logdet"}:
        jac = full_esr_transform_jacobian(mean)
        cov = ensure_spd(jac @ cov @ jac.T)
        criterion = "trace" if criterion == "esr_trace" else "logdet"
    if criterion == "trace":
        return float(np.trace(cov))
    if criterion == "logdet":
        return logdet_spd(cov)
    raise ValueError(f"unknown criterion {criterion}")


def ci_at_omega(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray, omega: float) -> FusionOutput:
    ya = inv_spd(cov_a)
    yb = inv_spd(cov_b)
    info = omega * ya + (1.0 - omega) * yb
    cov = inv_spd(info)
    mean = cov @ (omega * ya @ mean_a + (1.0 - omega) * yb @ mean_b)
    return FusionOutput(mean=mean, covariance=cov, omega=float(omega))


def ici_at_omega(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray, omega: float) -> FusionOutput:
    ya = inv_spd(cov_a)
    yb = inv_spd(cov_b)
    common_bound = ensure_spd(omega * cov_a + (1.0 - omega) * cov_b)
    yc = inv_spd(common_bound)
    info = ya + yb - yc
    valid = True
    try:
        np.linalg.cholesky(symmetrize(info))
    except np.linalg.LinAlgError:
        valid = False
    cov = inv_spd(info)
    mean_common = omega * mean_a + (1.0 - omega) * mean_b
    mean = cov @ (ya @ mean_a + yb @ mean_b - yc @ mean_common)
    return FusionOutput(mean=mean, covariance=cov, valid=valid, omega=float(omega))


def ci_or_ici_fusion(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    method: str,
    criterion: str,
    grid_size: int,
) -> FusionOutput:
    grid = np.linspace(0.0, 1.0, max(2, int(grid_size)))
    best: Optional[Tuple[float, FusionOutput]] = None
    for omega in grid:
        out = ci_at_omega(mean_a, cov_a, mean_b, cov_b, omega) if method == "ci" else ici_at_omega(mean_a, cov_a, mean_b, cov_b, omega)
        obj = covariance_objective(out.covariance, out.mean, criterion)
        if not out.valid:
            obj += 1e6
        if best is None or obj < best[0]:
            best = (obj, out)
    assert best is not None
    return best[1]


def chernoff_log_normalizer(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray, omega: float) -> float:
    ya = inv_spd(cov_a)
    yb = inv_spd(cov_b)
    info = omega * ya + (1.0 - omega) * yb
    cov_f = inv_spd(info)
    y_f = omega * ya @ mean_a + (1.0 - omega) * yb @ mean_b
    quad_inputs = omega * float(mean_a.T @ ya @ mean_a) + (1.0 - omega) * float(mean_b.T @ yb @ mean_b)
    quad_fused = float(y_f.T @ cov_f @ y_f)
    weighted_logdet = omega * logdet_spd(cov_a) + (1.0 - omega) * logdet_spd(cov_b)
    return float(0.5 * logdet_spd(cov_f) - 0.5 * weighted_logdet + 0.5 * quad_fused - 0.5 * quad_inputs)


def red_gci_esr_fusion(mean_a: np.ndarray, cov_a: np.ndarray, mean_b_observed: np.ndarray, cov_b_observed: np.ndarray, grid_size: int) -> FusionOutput:
    """All-chart RED-GCI/ESR approximation.

    We fuse A with all four equivalent B charts using CI with an ESR-trace
    omega objective.  Pair weights are Chernoff normalizers.  The point estimate
    is obtained by moment matching the kinematic entries in Euclidean space and
    the shape entries in RED square-root space, mirroring the MMGW/ESR idea.
    """
    components: List[FusionOutput] = []
    log_weights: List[float] = []
    for k in range(4):
        mb, cb, _ = transform_chart(mean_b_observed, cov_b_observed, k)
        out = ci_or_ici_fusion(mean_a, cov_a, mb, cb, method="ci", criterion="esr_trace", grid_size=grid_size)
        components.append(out)
        log_weights.append(chernoff_log_normalizer(mean_a, cov_a, mb, cb, out.omega))

    lw = np.asarray(log_weights)
    lw -= logsumexp(lw)
    weights = np.exp(lw)

    mean = np.zeros(STATE_DIM)
    mean[:4] = sum(weights[i] * components[i].mean[:4] for i in range(len(components)))
    sqrt_mean = sum(weights[i] * shape_sqrt_params(components[i].mean[[AL, L, W]]) for i in range(len(components)))
    mean[[AL, L, W]] = sqrt_params_to_shape(sqrt_mean)

    # Conservative-ish moment match in the aligned chart of the final mean.
    cov = np.zeros((STATE_DIM, STATE_DIM))
    for wi, comp in zip(weights, components):
        aligned_mean, aligned_cov, _, _ = align_to_reference(comp.mean, comp.covariance, mean)
        diff = aligned_mean - mean
        diff[AL] = angle_diff(aligned_mean[AL], mean[AL])
        cov += wi * (aligned_cov + np.outer(diff, diff))
    return FusionOutput(mean=mean, covariance=ensure_spd(cov), omega=float(np.sum(weights * [c.omega for c in components])))


def make_base_matrices(rho: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Base uncertainties are deliberately anisotropic so that sensor A and B
    # provide complementary information about axes/position.
    p0_base = np.diag([2.5, 2.5, 1.0, 1.0, 0.9, 1.0, 0.6]) ** 2
    r_a = np.diag([0.7, 1.4, 0.35, 0.55, 0.22, 0.75, 0.18]) ** 2
    r_b = np.diag([1.4, 0.7, 0.55, 0.35, 0.22, 0.18, 0.75]) ** 2

    # rho controls the common-prior information relative to local independent
    # information.  rho=0 is almost independent; rho≈1 is strongly correlated.
    rho = min(max(float(rho), 0.0), 0.999)
    common_strength = max(rho / max(1.0 - rho, 1e-6), 1e-6)
    y0 = common_strength * inv_spd(p0_base)
    ia = inv_spd(r_a)
    ib = inv_spd(r_b)
    p0 = inv_spd(y0)
    pa = inv_spd(y0 + ia)
    pb = inv_spd(y0 + ib)
    cross_ab = pa @ y0 @ pb
    return y0, p0, ia, ib, cross_ab


def sample_trial(rng: np.random.Generator, rho: float, chart_mode: str, fixed_chart: Optional[int] = None, axis_ratio: float = 3.0) -> TrialInputs:
    true_state = np.array([
        0.0,
        0.0,
        5.0,
        -1.0,
        rng.uniform(-np.pi, np.pi),
        axis_ratio,
        1.0,
    ])

    y0, p0, ia, ib, cross_ab = make_base_matrices(rho)
    ra = inv_spd(ia)
    rb = inv_spd(ib)

    common_mean = rng.multivariate_normal(true_state, p0)
    meas_a = rng.multivariate_normal(true_state, ra)
    meas_b = rng.multivariate_normal(true_state, rb)
    meas_a[L] = max(abs(meas_a[L]), 0.1)
    meas_a[W] = max(abs(meas_a[W]), 0.1)
    meas_b[L] = max(abs(meas_b[L]), 0.1)
    meas_b[W] = max(abs(meas_b[W]), 0.1)

    cov_a = inv_spd(y0 + ia)
    cov_b = inv_spd(y0 + ib)
    est_a = cov_a @ (y0 @ common_mean + ia @ meas_a)
    est_b = cov_b @ (y0 @ common_mean + ib @ meas_b)
    est_a[L] = max(abs(est_a[L]), 0.1)
    est_a[W] = max(abs(est_a[W]), 0.1)
    est_b[L] = max(abs(est_b[L]), 0.1)
    est_b[W] = max(abs(est_b[W]), 0.1)

    if chart_mode == "none":
        chart_b = 0
    elif chart_mode == "fixed":
        chart_b = int(fixed_chart or 0) % 4
    elif chart_mode == "random":
        chart_b = int(rng.integers(0, 4))
    else:
        raise ValueError(f"unknown chart_mode={chart_mode}")

    est_b_obs, cov_b_obs, cross_obs = transform_chart(est_b, cov_b, chart_b, cross_ab, transform_side="b")
    common_mean[L] = max(abs(common_mean[L]), 0.1)
    common_mean[W] = max(abs(common_mean[W]), 0.1)

    return TrialInputs(
        true_state=true_state,
        common_mean=common_mean,
        common_cov=p0,
        est_a=est_a,
        cov_a=cov_a,
        est_b_observed=est_b_obs,
        cov_b_observed=cov_b_obs,
        cross_ab_observed=cross_obs if cross_obs is not None else cross_ab,
        chart_b=chart_b,
    )


def fuse_method(method: str, trial: TrialInputs, gamma_wrong: float, omega_grid_size: int) -> FusionOutput:
    # RED-aware methods first align the incoming B chart with A.  Parameter-CI
    # intentionally does not align, so it exposes why CI alone is insufficient.
    b_aligned, cov_b_aligned, cross_aligned, _ = align_to_reference(
        trial.est_b_observed,
        trial.cov_b_observed,
        trial.est_a,
        cross_ab=trial.cross_ab_observed,
        transform_side="b",
    )
    common_aligned, common_cov_aligned, _, _ = align_to_reference(
        trial.common_mean,
        trial.common_cov,
        trial.est_a,
        cross_ab=None,
        transform_side="b",
    )

    if method == "oracle_correlated_red":
        return oracle_correlated_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned, cross_aligned)
    if method == "red_normal":
        return independent_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned)
    if method == "red_if_correct":
        return red_if_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned, common_aligned, common_cov_aligned, gamma=1.0)
    if method == "red_if_wrong":
        return red_if_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned, common_aligned, common_cov_aligned, gamma=gamma_wrong)
    if method == "parameter_ci":
        return ci_or_ici_fusion(
            trial.est_a,
            trial.cov_a,
            trial.est_b_observed,
            trial.cov_b_observed,
            method="ci",
            criterion="logdet",
            grid_size=omega_grid_size,
        )
    if method == "red_ci":
        return ci_or_ici_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned, method="ci", criterion="logdet", grid_size=omega_grid_size)
    if method == "red_ici":
        return ci_or_ici_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned, method="ici", criterion="logdet", grid_size=omega_grid_size)
    if method == "red_ci_esr":
        return ci_or_ici_fusion(trial.est_a, trial.cov_a, b_aligned, cov_b_aligned, method="ci", criterion="esr_trace", grid_size=omega_grid_size)
    if method == "red_gci_esr":
        return red_gci_esr_fusion(trial.est_a, trial.cov_a, trial.est_b_observed, trial.cov_b_observed, grid_size=omega_grid_size)
    raise ValueError(method)


def run_point(
    *,
    suite: str,
    methods: Sequence[str],
    trials: int,
    rho: float,
    gamma_wrong: float,
    chart_mode: str,
    fixed_chart: Optional[int],
    axis_ratio: float,
    seed: int,
    omega_grid_size: int,
) -> List[Dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    accs = {m: MetricAccumulator([], [], [], [], []) for m in methods}

    for _ in range(trials):
        trial = sample_trial(rng, rho=rho, chart_mode=chart_mode, fixed_chart=fixed_chart, axis_ratio=axis_ratio)
        for method in methods:
            start = time.perf_counter()
            try:
                out = fuse_method(method, trial, gamma_wrong=gamma_wrong, omega_grid_size=omega_grid_size)
            except Exception:
                runtime_ms = (time.perf_counter() - start) * 1000.0
                accs[method].invalid += 1
                accs[method].gw.append(np.nan)
                accs[method].iou.append(np.nan)
                accs[method].nees.append(np.nan)
                accs[method].logdet.append(np.nan)
                accs[method].runtime_ms.append(runtime_ms)
                continue
            runtime_ms = (time.perf_counter() - start) * 1000.0
            accs[method].append(out, trial.true_state, runtime_ms)

    rows: List[Dict[str, float | int | str]] = []
    lower = chi2.ppf(0.025, STATE_DIM * trials) / trials
    upper = chi2.ppf(0.975, STATE_DIM * trials) / trials
    for method in methods:
        summ = accs[method].summary()
        rows.append({
            "suite": suite,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "rho": rho,
            "gamma_wrong": gamma_wrong,
            "chart_mode": chart_mode,
            "fixed_chart": -1 if fixed_chart is None else int(fixed_chart),
            "axis_ratio": axis_ratio,
            "trials": trials,
            "nees_lower_95": lower,
            "nees_upper_95": upper,
            **summ,
        })
    return rows


def run_suite(args: argparse.Namespace) -> List[Dict[str, float | int | str]]:
    methods = parse_method_list(args.methods)
    rho_values = parse_float_list(args.rho_values)
    gamma_values = parse_float_list(args.gamma_values)
    rows: List[Dict[str, float | int | str]] = []

    if args.suite in {"correlation", "all"}:
        for idx, rho in enumerate(rho_values):
            rows.extend(run_point(
                suite="correlation",
                methods=methods,
                trials=args.trials,
                rho=rho,
                gamma_wrong=args.gamma_wrong_default,
                chart_mode="random",
                fixed_chart=None,
                axis_ratio=args.axis_ratio,
                seed=args.seed + 1000 * idx,
                omega_grid_size=args.omega_grid_size,
            ))

    if args.suite in {"wrong_common_info", "all"}:
        for idx, gamma in enumerate(gamma_values):
            rows.extend(run_point(
                suite="wrong_common_info",
                methods=methods,
                trials=args.trials,
                rho=args.rho_for_mismatch,
                gamma_wrong=gamma,
                chart_mode="random",
                fixed_chart=None,
                axis_ratio=args.axis_ratio,
                seed=args.seed + 2000 + 1000 * idx,
                omega_grid_size=args.omega_grid_size,
            ))

    if args.suite in {"ambiguity", "all"}:
        for idx, chart in enumerate([0, 1, 2, 3]):
            rows.extend(run_point(
                suite="ambiguity",
                methods=methods,
                trials=args.trials,
                rho=args.rho_for_ambiguity,
                gamma_wrong=args.gamma_wrong_default,
                chart_mode="fixed",
                fixed_chart=chart,
                axis_ratio=args.axis_ratio,
                seed=args.seed + 4000 + 1000 * idx,
                omega_grid_size=args.omega_grid_size,
            ))
    return rows


def write_csv(path: str, rows: List[Dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_order(method: str) -> int:
    return DEFAULT_METHODS.index(method) if method in DEFAULT_METHODS else len(DEFAULT_METHODS)


def write_summary(path: str, rows: List[Dict[str, float | int | str]]) -> None:
    with open(path, "w") as f:
        f.write("## Controlled unknown-correlation benchmark\n\n")
        f.write("Methods included: " + ", ".join(DEFAULT_METHODS) + "\n\n")
        for suite in ["correlation", "wrong_common_info", "ambiguity"]:
            suite_rows = [r for r in rows if r["suite"] == suite]
            if not suite_rows:
                continue
            f.write(f"### {suite.replace('_', ' ').title()}\n\n")
            # Compact last/most severe point table.
            if suite == "correlation":
                selector = max(float(r["rho"]) for r in suite_rows)
                selected = [r for r in suite_rows if float(r["rho"]) == selector]
                f.write(f"Worst/highest-correlation point: rho={selector:.2f}\n\n")
            elif suite == "wrong_common_info":
                selector = max(float(r["gamma_wrong"]) for r in suite_rows)
                selected = [r for r in suite_rows if float(r["gamma_wrong"]) == selector]
                f.write(f"Largest wrong common-info scale: gamma={selector:.2f}\n\n")
            else:
                selected = [r for r in suite_rows if int(r["fixed_chart"]) == 1]
                f.write("Adversarial equivalent-chart point: chart=1, i.e., alpha+pi/2 and axes swapped.\n\n")
            f.write("| Method | GW ↓ | IoU ↑ | NEES | NEES ratio | Invalid | Runtime ms |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in sorted(selected, key=lambda row: method_order(str(row["method"]))):
                f.write(
                    f"| {r['label']} | {float(r['mean_gw']):.4g} | {float(r['mean_iou']):.4g} | "
                    f"{float(r['mean_nees']):.4g} | {float(r['nees_ratio']):.3f} | "
                    f"{float(r['invalid_rate']):.3f} | {float(r['mean_runtime_ms']):.3f} |\n"
                )
            f.write("\n")


def plot_suite(rows: List[Dict[str, float | int | str]], suite: str, x_key: str, out_dir: str) -> None:
    suite_rows = [r for r in rows if r["suite"] == suite]
    if not suite_rows:
        return
    methods = sorted({str(r["method"]) for r in suite_rows}, key=method_order)
    for metric, ylabel in [("mean_gw", "mean squared GW"), ("mean_nees", "mean NEES")]:
        plt.figure(figsize=(8, 5))
        for method in methods:
            mr = sorted([r for r in suite_rows if r["method"] == method], key=lambda row: float(row[x_key]))
            plt.plot([float(r[x_key]) for r in mr], [float(r[metric]) for r in mr], marker="o", label=method)
        plt.xlabel(x_key)
        plt.ylabel(ylabel)
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{suite}_{metric}_vs_{x_key}.png"), dpi=150)
        plt.close()


def plot_ambiguity(rows: List[Dict[str, float | int | str]], out_dir: str) -> None:
    suite_rows = [r for r in rows if r["suite"] == "ambiguity"]
    if not suite_rows:
        return
    methods = sorted({str(r["method"]) for r in suite_rows}, key=method_order)
    plt.figure(figsize=(8, 5))
    width = 0.8 / max(len(methods), 1)
    charts = [0, 1, 2, 3]
    for idx, method in enumerate(methods):
        vals = []
        for chart in charts:
            match = [r for r in suite_rows if r["method"] == method and int(r["fixed_chart"]) == chart]
            vals.append(float(match[0]["mean_gw"]) if match else np.nan)
        xs = np.arange(len(charts)) + idx * width - 0.4 + width / 2.0
        plt.bar(xs, vals, width=width, label=method)
    plt.xticks(np.arange(len(charts)), [str(c) for c in charts])
    plt.xlabel("B equivalent RED chart")
    plt.ylabel("mean squared GW")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ambiguity_mean_gw_by_chart.png"), dpi=150)
    plt.close()


def make_plots(rows: List[Dict[str, float | int | str]], out_dir: str) -> None:
    plot_suite(rows, "correlation", "rho", out_dir)
    plot_suite(rows, "wrong_common_info", "gamma_wrong", out_dir)
    plot_ambiguity(rows, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["correlation", "wrong_common_info", "ambiguity", "all"], default="all")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rho-values", default="0.00,0.25,0.50,0.75,0.90,0.99")
    parser.add_argument("--gamma-values", default="0.00,0.25,0.50,1.00,2.00,4.00")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--omega-grid-size", type=int, default=31)
    parser.add_argument("--rho-for-mismatch", type=float, default=0.75)
    parser.add_argument("--rho-for-ambiguity", type=float, default=0.50)
    parser.add_argument("--gamma-wrong-default", type=float, default=0.50)
    parser.add_argument("--axis-ratio", type=float, default=5.0)
    parser.add_argument("--out-dir", default="evaluation_results/controlled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = run_suite(args)
    csv_path = os.path.join(args.out_dir, "controlled_unknown_corr_metrics.csv")
    write_csv(csv_path, rows)
    write_summary(os.path.join(args.out_dir, "summary.md"), rows)
    make_plots(rows, args.out_dir)
    print(f"Wrote {len(rows)} result rows to {csv_path}")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
