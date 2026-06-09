"""Unknown-correlation fusion for RED ellipse tracks.

This module is meant as an add-on to
https://github.com/Fusion-Goettingen/Fusion_2022_Thormann_RED-IF.

It implements conservative track-to-track fusion for elliptical extended
object estimates parameterized by orientation and semi-axis lengths.  The
main entry point is :class:`UnknownCorrelationFusionCenter`, which keeps
the RED/MMGW machinery from the original repository but replaces the
Kalman-style product update, and the RED-IF duplicate-information update,
by component-wise covariance intersection (CI).  Experimental inverse
covariance intersection (ICI), covariance-union safety inflation (CU),
Chernoff/GCI component weighting, and ESR-geometry-aware CI weight
selection are also provided.

The important assumption is different from the original RED-IF paper:
we do not know the cross-covariance or exact common information between
tracks.  Therefore, the fusion is intentionally conservative.  RED-IF is
expected to be sharper when its known-common-information assumptions are
correct; RED-CI/RED-GCI are fallbacks for the unknown-correlation case; RED-CI-CU adds a conservative safety valve for strongly inconsistent tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.special import logsumexp as lse

from Filters.fusioncenter import FusionCenter
from Filters.filtersupport import (
    mmgw_estimate_from_particles,
    reduce_mult_salmond,
    sample_mult,
    turn_mult,
)
from constants import *

UnknownCorrelationMethod = Literal["ci", "ici", "cu", "ci_cu"]
OmegaCriterion = Literal["logdet", "trace", "esr_trace", "esr_logdet"]
ComponentWeightMode = Literal["likelihood", "esr_likelihood", "chernoff", "prior", "uniform"]
ComponentPairingMode = Literal["all", "best", "gated"]


@dataclass(frozen=True)
class FusionResult:
    """Return object for one Gaussian unknown-correlation fusion."""

    mean: np.ndarray
    covariance: np.ndarray
    omega: float


@dataclass(frozen=True)
class _ShapeCandidate:
    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float
    score: float
    prior_id: int
    meas_id: int
    omega: float


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _ensure_spd(mat: np.ndarray, min_jitter: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """Return a symmetric positive-definite copy of ``mat``.

    Numerical roundoff or component reduction can create tiny negative
    eigenvalues.  This function adds diagonal jitter only when necessary.
    """

    out = _symmetrize(np.asarray(mat, dtype=float))
    eye = np.eye(out.shape[0])
    jitter = min_jitter
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(out)
            return out
        except np.linalg.LinAlgError:
            out = _symmetrize(out + jitter * eye)
            jitter *= 10.0
    # Last resort: eigenvalue clipping. This should be rare and is preferable
    # to crashing during long Monte Carlo sweeps.
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, min_jitter)
    return _symmetrize(vecs @ np.diag(vals) @ vecs.T)


def _safe_inv(mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_ensure_spd(mat))


def _logdet_spd(mat: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(_ensure_spd(mat))
    if sign <= 0:
        return np.inf
    return float(value)


def shape_to_sqrt_params(shape_mean: np.ndarray) -> np.ndarray:
    """Map [alpha, length, width] to symmetric square-root-matrix entries.

    This is the RED/MMGW square-root-space transform used in the original
    RED papers, but without the center and kinematic entries:

        S = R(alpha) diag(length, width) R(alpha)^T
        return [S_11, S_12, S_22]

    The transform is invariant to the equivalent RED chart changes
    [alpha, l, w] -> [alpha + pi/2, w, l] and [alpha + pi, l, w].
    """

    alpha, length, width = np.asarray(shape_mean, dtype=float)
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([
        length * c * c + width * s * s,
        (length - width) * s * c,
        length * s * s + width * c * c,
    ])


def shape_sqrt_jacobian(shape_mean: np.ndarray) -> np.ndarray:
    """Jacobian of :func:`shape_to_sqrt_params` at [alpha, length, width]."""

    alpha, length, width = np.asarray(shape_mean, dtype=float)
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([
        [2.0 * s * c * (width - length), c * c, s * s],
        [(length - width) * (c * c - s * s), s * c, -s * c],
        [2.0 * s * c * (length - width), s * s, c * c],
    ])


def _objective(cov: np.ndarray, criterion: OmegaCriterion, mean: Optional[np.ndarray] = None) -> float:
    """Covariance-size objective for CI/ICI omega selection.

    ``esr_trace`` and ``esr_logdet`` evaluate shape uncertainty after the
    linearized RED square-root transform.  They are only meaningful for the
    3-D shape vector [alpha, length, width].  If called for another state
    dimension, the function falls back to the matching raw covariance
    objective, making it safe to reuse for the kinematic state.
    """

    cov = _ensure_spd(cov)
    if criterion in ("esr_trace", "esr_logdet"):
        if mean is not None and len(np.atleast_1d(mean)) == 3:
            jac = shape_sqrt_jacobian(mean)
            cov = _ensure_spd(jac @ cov @ jac.T)
        criterion = "trace" if criterion == "esr_trace" else "logdet"

    if criterion == "trace":
        return float(np.trace(cov))
    if criterion == "logdet":
        return _logdet_spd(cov)
    raise ValueError(f"Unknown omega criterion: {criterion}")



def _local_chart_diff(mean: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Difference in the current local Gaussian chart.

    For shape vectors, the orientation component is wrapped.  The function is
    deliberately conservative for all other vector dimensions: it assumes the
    caller has already aligned equivalent RED charts.
    """

    diff = np.asarray(mean, dtype=float) - np.asarray(reference, dtype=float)
    if diff.shape[0] == 3:
        diff[0] = ((diff[0] + np.pi) % (2.0 * np.pi)) - np.pi
    return diff


def _compatibility_nis(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    esr: bool = False,
) -> float:
    """Return a normalized discrepancy between two aligned Gaussian estimates.

    For 3-D shape states and ``esr=True``, the discrepancy is measured in the
    RED square-root shape space.  This makes the gate insensitive to equivalent
    ellipse charts and less brittle near almost circular extents.
    """

    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)

    if esr and mean_a.shape[0] == 3:
        diff = shape_to_sqrt_params(mean_b) - shape_to_sqrt_params(mean_a)
        jac_a = shape_sqrt_jacobian(mean_a)
        jac_b = shape_sqrt_jacobian(mean_b)
        cov = _ensure_spd(jac_a @ cov_a @ jac_a.T + jac_b @ cov_b @ jac_b.T)
    else:
        diff = _local_chart_diff(mean_b, mean_a)
        cov = _ensure_spd(cov_a + cov_b)
    return float(diff.T @ _safe_inv(cov) @ diff)


def _default_cu_gate_threshold(dim: int) -> float:
    """Chi-square-like default gate without depending on scipy.stats.

    The Wilson-Hilferty approximation here is intentionally loose.  It is only
    used as a safety switch for CI-CU, not as a formal hypothesis test.
    """

    dim = max(int(dim), 1)
    return float(dim * (1.0 - 2.0 / (9.0 * dim) + 1.96 * np.sqrt(2.0 / (9.0 * dim))) ** 3)


def covariance_union_inflate(
    mean_f: np.ndarray,
    cov_f: np.ndarray,
    inputs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    margin: float = 1e-6,
) -> np.ndarray:
    """Inflate a fused covariance so it covers all input Gaussian components.

    This is a practical covariance-union-style safety step.  For a fixed fused
    mean, each input component is represented by the second moment around that
    fused mean,

        C_i + (m_i - m_f)(m_i - m_f)^T.

    We then scale ``cov_f`` by the smallest scalar found by a generalized
    eigenvalue bound so that the scaled covariance dominates every such second
    moment.  It is not the minimum-determinant covariance union over all means,
    but it is simple, deterministic, and useful for inconsistent track pairs.
    """

    mean_f = np.asarray(mean_f, dtype=float)
    cov_f = _ensure_spd(cov_f)
    chol = np.linalg.cholesky(cov_f)
    scale = 1.0
    for mean_i, cov_i in inputs:
        diff = _local_chart_diff(np.asarray(mean_i, dtype=float), mean_f)
        cover = _ensure_spd(cov_i + np.outer(diff, diff))
        # Compute eigenvalues of C_f^{-1/2} cover C_f^{-T/2} without forming
        # an explicit inverse.
        whitened = np.linalg.solve(chol, cover)
        whitened = np.linalg.solve(chol, whitened.T).T
        eig_max = float(np.max(np.linalg.eigvalsh(_symmetrize(whitened))))
        if np.isfinite(eig_max):
            scale = max(scale, eig_max)
    return _ensure_spd((1.0 + float(margin)) * scale * cov_f)


def _ci_at_omega(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    omega: float,
) -> FusionResult:
    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)
    inv_a = _safe_inv(cov_a)
    inv_b = _safe_inv(cov_b)

    info = omega * inv_a + (1.0 - omega) * inv_b
    cov_f = _safe_inv(info)
    mean_f = cov_f @ (omega * inv_a @ mean_a + (1.0 - omega) * inv_b @ mean_b)
    return FusionResult(mean=mean_f, covariance=_ensure_spd(cov_f), omega=float(omega))


def _ici_at_omega(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    omega: float,
) -> FusionResult:
    """Inverse covariance intersection for one aligned Gaussian pair.

    ICI is included as an experimental sharper alternative.  It is less
    conservative than CI under the common-information model used in the ICI
    literature, but CI should remain the default baseline for unknown RED
    track correlations.
    """

    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)
    inv_a = _safe_inv(cov_a)
    inv_b = _safe_inv(cov_b)

    common_bound = _ensure_spd(omega * cov_a + (1.0 - omega) * cov_b)
    inv_common_bound = _safe_inv(common_bound)

    info = inv_a + inv_b - inv_common_bound
    cov_f = _safe_inv(info)
    mean_common = omega * mean_a + (1.0 - omega) * mean_b
    mean_f = cov_f @ (inv_a @ mean_a + inv_b @ mean_b - inv_common_bound @ mean_common)
    return FusionResult(mean=mean_f, covariance=_ensure_spd(cov_f), omega=float(omega))


def chernoff_log_normalizer(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    omega: float,
) -> float:
    """Log integral of N_a(x)^omega N_b(x)^(1-omega).

    For Gaussian components, covariance intersection is equivalent to a
    Chernoff/geometric-mean density fusion.  The normalization coefficient is
    useful as a principled RED mixture-pair weight: components that disagree
    receive a lower coefficient without assuming statistical independence.
    """

    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)
    inv_a = _safe_inv(cov_a)
    inv_b = _safe_inv(cov_b)

    info = omega * inv_a + (1.0 - omega) * inv_b
    cov_f = _safe_inv(info)
    y_f = omega * inv_a @ mean_a + (1.0 - omega) * inv_b @ mean_b

    quad_inputs = (
        omega * float(mean_a.T @ inv_a @ mean_a)
        + (1.0 - omega) * float(mean_b.T @ inv_b @ mean_b)
    )
    quad_fused = float(y_f.T @ cov_f @ y_f)
    weighted_logdet = omega * _logdet_spd(cov_a) + (1.0 - omega) * _logdet_spd(cov_b)
    return float(0.5 * _logdet_spd(cov_f) - 0.5 * weighted_logdet + 0.5 * quad_fused - 0.5 * quad_inputs)


def fuse_gaussians_unknown_correlation(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    method: UnknownCorrelationMethod = "ci",
    criterion: OmegaCriterion = "logdet",
    fixed_omega: Optional[float] = None,
    grid_size: int = 31,
    cu_gate_threshold: Optional[float] = None,
    cu_inflation_margin: float = 1e-6,
    cu_esr_gate: bool = True,
) -> FusionResult:
    """Fuse two Gaussian estimates without using a cross-covariance.

    Parameters
    ----------
    mean_a, cov_a:
        First Gaussian estimate.
    mean_b, cov_b:
        Second Gaussian estimate, expressed in the same local chart as the
        first one.  For REDs, use :func:`align_shape_mean_to_reference` before
        calling this function.
    method:
        ``"ci"`` for covariance intersection, ``"ici"`` for inverse
        covariance intersection, ``"cu"`` for unconditional covariance-union
        inflation around the CI mean, or ``"ci_cu"`` for CI with a
        covariance-union safety fallback only when the inputs disagree.
    criterion:
        Objective used for choosing omega: covariance log determinant, trace,
        or a linearized ESR/square-root-space objective for 3-D shape states.
    fixed_omega:
        If provided, bypasses the grid search.  ``0.5`` is a useful fast
        diagnostic setting.
    grid_size:
        Number of grid points in [0, 1] for omega selection when
        ``fixed_omega`` is not set.
    cu_gate_threshold:
        Gate for the CI-CU safety switch.  If omitted, a loose
        chi-square-like threshold based on the state dimension is used.
    cu_inflation_margin:
        Extra multiplicative covariance inflation used by the covariance-union
        safety step.
    cu_esr_gate:
        Use square-root ellipse geometry for the shape compatibility gate.
    """

    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)

    if method not in ("ci", "ici", "cu", "ci_cu"):
        raise ValueError(f"Unknown unknown-correlation fusion method: {method}")

    # CU and CI-CU use CI as the nominal estimate, then inflate if needed.
    fuse_at = _ici_at_omega if method == "ici" else _ci_at_omega

    if fixed_omega is not None:
        omega = float(np.clip(fixed_omega, 0.0, 1.0))
        best_result: Optional[FusionResult] = fuse_at(mean_a, cov_a, mean_b, cov_b, omega)
    else:
        # A small grid is deliberate: this function is called for every RED
        # component pair in every Monte Carlo run.  Increase grid_size for final
        # experiments if runtime is acceptable.
        grid_size = max(int(grid_size), 2)
        candidates = np.linspace(0.0, 1.0, grid_size)
        best_value = np.inf
        best_result = None

        for omega in candidates:
            try:
                result = fuse_at(mean_a, cov_a, mean_b, cov_b, float(omega))
                value = _objective(result.covariance, criterion, result.mean)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
            if value < best_value:
                best_value = value
                best_result = result

        if best_result is None:
            # Very defensive fallback.  In practice this should not happen for CI
            # with SPD input covariances.
            best_result = _ci_at_omega(mean_a, cov_a, mean_b, cov_b, 0.5)

    if method in ("cu", "ci_cu"):
        nis = _compatibility_nis(
            mean_a, cov_a, mean_b, cov_b, esr=bool(cu_esr_gate)
        )
        gate = _default_cu_gate_threshold(mean_a.shape[0]) if cu_gate_threshold is None else float(cu_gate_threshold)
        if method == "cu" or nis > gate:
            best_result = FusionResult(
                mean=best_result.mean,
                covariance=covariance_union_inflate(
                    best_result.mean,
                    best_result.covariance,
                    [(mean_a, cov_a), (mean_b, cov_b)],
                    margin=cu_inflation_margin,
                ),
                omega=best_result.omega,
            )
    return best_result


def wrap_angle_to_reference(angle: float, reference: float) -> float:
    """Wrap ``angle`` so that the difference to ``reference`` is in [-pi, pi)."""

    return float(reference + ((angle - reference + np.pi) % (2.0 * np.pi)) - np.pi)


def align_shape_mean_to_reference(shape_mean: np.ndarray, reference_mean: np.ndarray) -> np.ndarray:
    """Align a RED shape component to a reference component's angle chart.

    The axis swap is already handled by ``turn_mult``.  This function only
    applies the 2*pi wrapping needed before Euclidean Gaussian fusion in the
    local chart.
    """

    aligned = np.asarray(shape_mean, dtype=float).copy()
    aligned[0] = wrap_angle_to_reference(aligned[0], float(reference_mean[0]))
    return aligned


def _log_gaussian_pdf(diff: np.ndarray, cov: np.ndarray) -> float:
    cov = _ensure_spd(cov)
    dim = diff.shape[0]
    return float(
        -0.5 * dim * np.log(2.0 * np.pi)
        -0.5 * _logdet_spd(cov)
        -0.5 * diff.T @ _safe_inv(cov) @ diff
    )


def _esr_log_gaussian_compatibility(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    scale: float = 1.0,
) -> float:
    """Approximate component compatibility in RED square-root space."""

    ta = shape_to_sqrt_params(mean_a)
    tb = shape_to_sqrt_params(mean_b)
    ja = shape_sqrt_jacobian(mean_a)
    jb = shape_sqrt_jacobian(mean_b)
    cov_t = scale * (ja @ _ensure_spd(cov_a) @ ja.T + jb @ _ensure_spd(cov_b) @ jb.T)
    return _log_gaussian_pdf(tb - ta, cov_t)


class UnknownCorrelationFusionCenter(FusionCenter):
    """Fusion center for RED track-to-track fusion under unknown correlation.

    This class keeps the original repository's RED representation and MMGW
    estimate, but uses CI/ICI when combining the current global track and each
    incoming local track.  It does not subtract duplicate information because
    the duplicate information is assumed unknown.
    """

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["use_if"] = False
        kwargs["use_red"] = True
        super().__init__(**kwargs)

        self._unknown_corr_method: UnknownCorrelationMethod = kwargs.get("unknown_corr_method", "ci")
        # Backward compatible: old scripts can keep using omega_criterion for both parts.
        self._omega_criterion: OmegaCriterion = kwargs.get("omega_criterion", "logdet")
        self._kin_omega_criterion: OmegaCriterion = kwargs.get("kin_omega_criterion", self._omega_criterion)
        self._shape_omega_criterion: OmegaCriterion = kwargs.get("shape_omega_criterion", self._omega_criterion)
        self._fixed_omega: Optional[float] = kwargs.get("fixed_omega", None)
        self._omega_grid_size: int = kwargs.get("omega_grid_size", 31)
        self._component_weight_mode: ComponentWeightMode = kwargs.get("component_weight_mode", "likelihood")
        self._component_pairing_mode: ComponentPairingMode = kwargs.get("component_pairing_mode", "all")
        self._component_gate_log_weight: float = float(kwargs.get("component_gate_log_weight", 12.0))
        self._compatibility_scale: float = float(kwargs.get("compatibility_scale", 1.0))
        self._cu_gate_threshold: Optional[float] = kwargs.get("cu_gate_threshold", None)
        self._cu_inflation_margin: float = float(kwargs.get("cu_inflation_margin", 1e-6))
        self._cu_esr_gate: bool = bool(kwargs.get("cu_esr_gate", True))
        self._estimate_samples: int = int(kwargs.get("estimate_samples", 1000))

        if self._component_pairing_mode not in ("all", "best", "gated"):
            raise ValueError(f"Unknown component_pairing_mode: {self._component_pairing_mode}")

        # The original constructor initializes a single Gaussian shape.  For an
        # unknown-correlation RED experiment, start directly in RED form.
        self.reset(kwargs["init_state"], kwargs["init_cov"])

    def correct(self, est: np.ndarray, est_cov: np.ndarray) -> None:
        """Fuse all incoming local track estimates at the current time."""

        for sensor_id in range(len(est)):
            self._correct_one_track(est[sensor_id], est_cov[sensor_id])
            self._shape_state, self._shape_cov, self._shape_weight = reduce_mult_salmond(
                self._shape_state, self._shape_cov, self._shape_weight
            )

        self._set_point_estimate()

    def _correct_one_track(self, est: np.ndarray, est_cov: np.ndarray) -> None:
        # Kinematic part: ordinary Gaussian CI/ICI in R^4.
        kin_result = fuse_gaussians_unknown_correlation(
            self._kin_state,
            self._kin_cov,
            est[[X1, X2, V1, V2]],
            est_cov[:4, :4],
            method=self._unknown_corr_method,
            criterion=self._kin_omega_criterion,
            fixed_omega=self._fixed_omega,
            grid_size=self._omega_grid_size,
            cu_gate_threshold=self._cu_gate_threshold,
            cu_inflation_margin=self._cu_inflation_margin,
            cu_esr_gate=False,
        )
        self._kin_state = kin_result.mean
        self._kin_cov = kin_result.covariance

        # Shape part: RED component alignment followed by component-wise CI/ICI.
        shape_est, shape_est_cov, shape_est_weight = turn_mult(est[[AL, L, W]], est_cov[4:, 4:])

        candidates = self._build_shape_candidates(shape_est, shape_est_cov, shape_est_weight)
        candidates = self._select_shape_candidates(candidates)

        if not candidates:
            # Defensive fallback: preserve the prior RED if all candidates were
            # numerically rejected.  This should not happen with sane input.
            return

        self._shape_state = np.array([c.mean for c in candidates])
        self._shape_cov = np.array([c.covariance for c in candidates])
        if self._component_weight_mode == "uniform":
            self._shape_weight = np.ones(len(candidates)) / len(candidates)
        else:
            log_w = np.array([c.log_weight for c in candidates])
            log_w -= lse(log_w)
            self._shape_weight = np.exp(log_w)

    def _build_shape_candidates(
        self,
        shape_est: np.ndarray,
        shape_est_cov: np.ndarray,
        shape_est_weight: np.ndarray,
    ) -> List[_ShapeCandidate]:
        n_prior = len(self._shape_weight)
        n_meas = len(shape_est_weight)
        prior_log_weight = np.log(np.maximum(self._shape_weight, np.finfo(float).tiny))
        meas_log_weight = np.log(np.maximum(shape_est_weight, np.finfo(float).tiny))

        candidates: List[_ShapeCandidate] = []
        for i in range(n_prior):
            prior_mean = self._shape_state[i]
            prior_cov = self._shape_cov[i]
            for j in range(n_meas):
                meas_mean = align_shape_mean_to_reference(shape_est[j], prior_mean)
                meas_cov = shape_est_cov[j]

                result = fuse_gaussians_unknown_correlation(
                    prior_mean,
                    prior_cov,
                    meas_mean,
                    meas_cov,
                    method=self._unknown_corr_method,
                    criterion=self._shape_omega_criterion,
                    fixed_omega=self._fixed_omega,
                    grid_size=self._omega_grid_size,
                    cu_gate_threshold=self._cu_gate_threshold,
                    cu_inflation_margin=self._cu_inflation_margin,
                    cu_esr_gate=self._cu_esr_gate,
                )
                log_weight = self._component_log_weight(
                    prior_log_weight[i],
                    meas_log_weight[j],
                    prior_mean,
                    prior_cov,
                    meas_mean,
                    meas_cov,
                    result.omega,
                )
                score = self._component_pairing_score(
                    prior_log_weight[i], meas_log_weight[j], prior_mean, prior_cov, meas_mean, meas_cov
                )
                candidates.append(_ShapeCandidate(
                    mean=result.mean,
                    covariance=result.covariance,
                    log_weight=log_weight,
                    score=score,
                    prior_id=i,
                    meas_id=j,
                    omega=result.omega,
                ))
        return candidates

    def _select_shape_candidates(self, candidates: List[_ShapeCandidate]) -> List[_ShapeCandidate]:
        if self._component_pairing_mode == "all" or len(candidates) <= 1:
            return candidates

        selected: List[_ShapeCandidate] = []
        prior_ids = sorted({c.prior_id for c in candidates})
        for prior_id in prior_ids:
            cur = [c for c in candidates if c.prior_id == prior_id]
            if not cur:
                continue
            best_score = max(c.score for c in cur)
            if self._component_pairing_mode == "best":
                selected.append(max(cur, key=lambda c: c.score))
            elif self._component_pairing_mode == "gated":
                keep = [c for c in cur if c.score >= best_score - self._component_gate_log_weight]
                selected.extend(keep if keep else [max(cur, key=lambda c: c.score)])
            else:
                raise ValueError(f"Unknown component_pairing_mode: {self._component_pairing_mode}")
        return selected

    def _component_log_weight(
        self,
        prior_log_weight: float,
        meas_log_weight: float,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        meas_mean: np.ndarray,
        meas_cov: np.ndarray,
        omega: float,
    ) -> float:
        if self._component_weight_mode == "uniform":
            return 0.0
        if self._component_weight_mode == "prior":
            return float(prior_log_weight + meas_log_weight)
        if self._component_weight_mode == "chernoff":
            if self._unknown_corr_method in ("ci", "cu", "ci_cu"):
                return float(
                    prior_log_weight
                    + meas_log_weight
                    + chernoff_log_normalizer(prior_mean, prior_cov, meas_mean, meas_cov, omega)
                )
            # ICI is not a Chernoff/geometric-mean density.  Use the geometry
            # compatibility fallback rather than silently pretending otherwise.
            return float(
                prior_log_weight
                + meas_log_weight
                + _esr_log_gaussian_compatibility(
                    prior_mean, prior_cov, meas_mean, meas_cov, self._compatibility_scale
                )
            )
        if self._component_weight_mode == "esr_likelihood":
            return float(
                prior_log_weight
                + meas_log_weight
                + _esr_log_gaussian_compatibility(
                    prior_mean, prior_cov, meas_mean, meas_cov, self._compatibility_scale
                )
            )
        if self._component_weight_mode != "likelihood":
            raise ValueError(f"Unknown component_weight_mode: {self._component_weight_mode}")

        diff = meas_mean - prior_mean
        diff[0] = ((diff[0] + np.pi) % (2.0 * np.pi)) - np.pi
        compat_cov = self._compatibility_scale * (prior_cov + meas_cov)
        return float(prior_log_weight + meas_log_weight + _log_gaussian_pdf(diff, compat_cov))

    def _component_pairing_score(
        self,
        prior_log_weight: float,
        meas_log_weight: float,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        meas_mean: np.ndarray,
        meas_cov: np.ndarray,
    ) -> float:
        """Geometry-aware score used only for best/gated RED chart selection."""

        return float(
            prior_log_weight
            + meas_log_weight
            + _esr_log_gaussian_compatibility(
                prior_mean, prior_cov, meas_mean, meas_cov, self._compatibility_scale
            )
        )

    def _set_point_estimate(self) -> None:
        if self._estimate_samples <= 0:
            shape_est = self._shape_state[np.argmax(self._shape_weight)]
        else:
            particles = sample_mult(self._shape_state, self._shape_cov, self._shape_weight, self._estimate_samples)
            shape_est = mmgw_estimate_from_particles(particles)
        self._est = np.hstack([self._kin_state, shape_est])


# Backwards-compatible aliases that make experiment scripts a little clearer.
RedCovarianceIntersectionFusionCenter = UnknownCorrelationFusionCenter
REDCIFusionCenter = UnknownCorrelationFusionCenter
