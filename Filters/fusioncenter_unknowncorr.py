"""Unknown-correlation fusion for RED ellipse tracks.

This module is meant as an add-on to
https://github.com/Fusion-Goettingen/Fusion_2022_Thormann_RED-IF.

It implements conservative track-to-track fusion for elliptical extended
object estimates parameterized by orientation and semi-axis lengths.  The
main entry point is :class:`UnknownCorrelationFusionCenter`, which keeps
the RED/MMGW machinery from the original repository but replaces the
Kalman-style product update, and the RED-IF duplicate-information update,
by component-wise covariance intersection (CI).  An experimental inverse
covariance intersection (ICI) option is also provided.

The important assumption is different from the original RED-IF paper:
we do not know the cross-covariance or exact common information between
tracks.  Therefore, the fusion is intentionally conservative.  RED-IF is
expected to be sharper when its known-common-information assumptions are
correct; RED-CI is a fallback for the unknown-correlation case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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

UnknownCorrelationMethod = Literal["ci", "ici"]
OmegaCriterion = Literal["logdet", "trace"]
ComponentWeightMode = Literal["likelihood", "prior", "uniform"]


@dataclass(frozen=True)
class FusionResult:
    """Return object for one Gaussian unknown-correlation fusion."""

    mean: np.ndarray
    covariance: np.ndarray
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


def _objective(cov: np.ndarray, criterion: OmegaCriterion) -> float:
    cov = _ensure_spd(cov)
    if criterion == "trace":
        return float(np.trace(cov))
    if criterion == "logdet":
        return _logdet_spd(cov)
    raise ValueError(f"Unknown omega criterion: {criterion}")


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
        ``"ci"`` for covariance intersection or ``"ici"`` for inverse
        covariance intersection.
    criterion:
        Objective used for choosing omega: covariance log determinant or trace.
    fixed_omega:
        If provided, bypasses the grid search.  ``0.5`` is a useful fast
        diagnostic setting.
    grid_size:
        Number of grid points in [0, 1] for omega selection when
        ``fixed_omega`` is not set.
    """

    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)

    if method not in ("ci", "ici"):
        raise ValueError(f"Unknown unknown-correlation fusion method: {method}")

    fuse_at = _ci_at_omega if method == "ci" else _ici_at_omega

    if fixed_omega is not None:
        omega = float(np.clip(fixed_omega, 0.0, 1.0))
        return fuse_at(mean_a, cov_a, mean_b, cov_b, omega)

    inv_a = _safe_inv(cov_a)
    inv_b = _safe_inv(cov_b)
    info_mean_a = inv_a @ mean_a
    info_mean_b = inv_b @ mean_b

    # A small grid is deliberate: this function is called for every RED
    # component pair in every Monte Carlo run.  Increase grid_size for final
    # experiments if runtime is acceptable.
    grid_size = max(int(grid_size), 2)
    candidates = np.linspace(0.0, 1.0, grid_size)
    best_omega: Optional[float] = None

    try:
        omega_weights = candidates[:, None, None]
        if method == "ci":
            info_grid = omega_weights * inv_a + (1.0 - omega_weights) * inv_b
        else:
            common_bound_grid = _symmetrize(omega_weights * cov_a + (1.0 - omega_weights) * cov_b)
            inv_common_bound_grid = np.linalg.inv(common_bound_grid)
            info_grid = inv_a + inv_b - inv_common_bound_grid

        if criterion == "logdet":
            signs, logdets = np.linalg.slogdet(info_grid)
            if method == "ici" and np.any(signs <= 0):
                raise np.linalg.LinAlgError("non-SPD ICI information matrix")
            values = np.where(signs > 0, -logdets, np.inf)
        elif criterion == "trace":
            cov_grid = np.linalg.inv(info_grid)
            values = np.trace(cov_grid, axis1=1, axis2=2)
        else:
            raise ValueError(f"Unknown omega criterion: {criterion}")

        if np.any(np.isfinite(values)):
            best_omega = float(candidates[int(np.nanargmin(values))])
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        best_value = np.inf
        for omega in candidates:
            try:
                if method == "ci":
                    info = omega * inv_a + (1.0 - omega) * inv_b
                else:
                    common_bound = _ensure_spd(omega * cov_a + (1.0 - omega) * cov_b)
                    inv_common_bound = _safe_inv(common_bound)
                    info = inv_a + inv_b - inv_common_bound

                if criterion == "logdet":
                    value = -_logdet_spd(info)
                elif criterion == "trace":
                    value = float(np.trace(_safe_inv(info)))
                else:
                    raise ValueError(f"Unknown omega criterion: {criterion}")
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
            if value < best_value:
                best_value = value
                best_omega = float(omega)

    if best_omega is None:
        # Very defensive fallback.  In practice this should not happen for CI
        # with SPD input covariances.
        return _ci_at_omega(mean_a, cov_a, mean_b, cov_b, 0.5)

    if method == "ci":
        info = best_omega * inv_a + (1.0 - best_omega) * inv_b
        cov_f = _safe_inv(info)
        mean_f = cov_f @ (best_omega * info_mean_a + (1.0 - best_omega) * info_mean_b)
        return FusionResult(mean=mean_f, covariance=_ensure_spd(cov_f), omega=best_omega)

    return fuse_at(mean_a, cov_a, mean_b, cov_b, best_omega)


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
        self._omega_criterion: OmegaCriterion = kwargs.get("omega_criterion", "logdet")
        self._fixed_omega: Optional[float] = kwargs.get("fixed_omega", None)
        self._omega_grid_size: int = kwargs.get("omega_grid_size", 31)
        self._component_weight_mode: ComponentWeightMode = kwargs.get("component_weight_mode", "likelihood")
        self._compatibility_scale: float = float(kwargs.get("compatibility_scale", 1.0))
        self._estimate_samples: int = int(kwargs.get("estimate_samples", 1000))

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
            criterion=self._omega_criterion,
            fixed_omega=self._fixed_omega,
            grid_size=self._omega_grid_size,
        )
        self._kin_state = kin_result.mean
        self._kin_cov = kin_result.covariance

        # Shape part: RED component alignment followed by component-wise CI/ICI.
        shape_est, shape_est_cov, shape_est_weight = turn_mult(est[[AL, L, W]], est_cov[4:, 4:])

        n_prior = len(self._shape_weight)
        n_meas = len(shape_est_weight)
        new_shape_state = np.zeros((n_prior * n_meas, 3))
        new_shape_cov = np.zeros((n_prior * n_meas, 3, 3))
        new_log_weight = np.zeros(n_prior * n_meas)

        prior_log_weight = np.log(np.maximum(self._shape_weight, np.finfo(float).tiny))
        meas_log_weight = np.log(np.maximum(shape_est_weight, np.finfo(float).tiny))

        out_id = 0
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
                    criterion=self._omega_criterion,
                    fixed_omega=self._fixed_omega,
                    grid_size=self._omega_grid_size,
                )
                new_shape_state[out_id] = result.mean
                new_shape_cov[out_id] = result.covariance
                new_log_weight[out_id] = self._component_log_weight(
                    prior_log_weight[i], meas_log_weight[j], prior_mean, prior_cov, meas_mean, meas_cov
                )
                out_id += 1

        if self._component_weight_mode == "uniform":
            self._shape_weight = np.ones_like(new_log_weight) / len(new_log_weight)
        else:
            new_log_weight -= lse(new_log_weight)
            self._shape_weight = np.exp(new_log_weight)
        self._shape_state = new_shape_state
        self._shape_cov = new_shape_cov

    def _component_log_weight(
        self,
        prior_log_weight: float,
        meas_log_weight: float,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        meas_mean: np.ndarray,
        meas_cov: np.ndarray,
    ) -> float:
        if self._component_weight_mode == "uniform":
            return 0.0
        if self._component_weight_mode == "prior":
            return float(prior_log_weight + meas_log_weight)
        if self._component_weight_mode != "likelihood":
            raise ValueError(f"Unknown component_weight_mode: {self._component_weight_mode}")

        diff = meas_mean - prior_mean
        diff[0] = ((diff[0] + np.pi) % (2.0 * np.pi)) - np.pi
        compat_cov = self._compatibility_scale * (prior_cov + meas_cov)
        return float(prior_log_weight + meas_log_weight + _log_gaussian_pdf(diff, compat_cov))

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
