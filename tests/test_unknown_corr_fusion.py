"""Small tests for the RED unknown-correlation add-on.

Run from the repository root after copying the patch files:

    python -m pytest tests/test_unknown_corr_fusion.py
"""

import numpy as np

from Filters.filtersupport import turn_mult
from Filters.fusioncenter_unknowncorr import (
    align_shape_mean_to_reference,
    chernoff_log_normalizer,
    covariance_union_inflate,
    fuse_gaussians_unknown_correlation,
    shape_sqrt_jacobian,
    shape_to_sqrt_params,
)


def test_ci_identical_estimates_do_not_double_count_information():
    mean = np.array([1.0, -2.0, 0.5])
    cov = np.diag([0.3, 2.0, 0.7])

    fused = fuse_gaussians_unknown_correlation(
        mean,
        cov,
        mean,
        cov,
        method="ci",
        fixed_omega=0.37,
    )

    np.testing.assert_allclose(fused.mean, mean, atol=1e-12)
    np.testing.assert_allclose(fused.covariance, cov, atol=1e-12)


def test_ordinary_independent_information_product_would_be_overconfident():
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)
    info_fused_cov = np.linalg.inv(np.linalg.inv(cov) + np.linalg.inv(cov))

    assert np.allclose(info_fused_cov, 0.5 * cov)

    ci_fused = fuse_gaussians_unknown_correlation(mean, cov, mean, cov, method="ci", fixed_omega=0.5)
    np.testing.assert_allclose(ci_fused.covariance, cov, atol=1e-12)


def test_red_alignment_finds_equivalent_axis_swapped_representation():
    # These two parameter vectors describe the same physical ellipse:
    # [alpha=0, a=5, b=1.5] and [alpha=pi/2, a=1.5, b=5].
    reference = np.array([0.0, 5.0, 1.5])
    equivalent = np.array([0.5 * np.pi, 1.5, 5.0])
    cov = np.diag([0.05, 0.2, 0.2])

    red_means, _, _ = turn_mult(equivalent, cov)
    aligned = np.array([align_shape_mean_to_reference(component, reference) for component in red_means])
    distances = np.linalg.norm(aligned - reference, axis=1)

    assert np.min(distances) < 1e-10


def test_square_root_shape_transform_is_red_invariant():
    reference = np.array([0.3, 6.0, 2.0])
    equivalent_a = np.array([0.3 + np.pi, 6.0, 2.0])
    equivalent_b = np.array([0.3 + 0.5 * np.pi, 2.0, 6.0])

    t_ref = shape_to_sqrt_params(reference)
    np.testing.assert_allclose(shape_to_sqrt_params(equivalent_a), t_ref, atol=1e-12)
    np.testing.assert_allclose(shape_to_sqrt_params(equivalent_b), t_ref, atol=1e-12)


def test_square_root_shape_jacobian_matches_finite_difference():
    mean = np.array([0.4, 5.0, 1.7])
    jac = shape_sqrt_jacobian(mean)
    eps = 1e-6
    fd = np.zeros((3, 3))
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        fd[:, i] = (shape_to_sqrt_params(mean + step) - shape_to_sqrt_params(mean - step)) / (2.0 * eps)
    np.testing.assert_allclose(jac, fd, atol=1e-6)


def test_chernoff_log_normalizer_is_zero_for_identical_gaussians():
    mean = np.array([0.3, 5.0, 2.0])
    cov = np.diag([0.2, 0.5, 0.4])
    assert abs(chernoff_log_normalizer(mean, cov, mean, cov, 0.23)) < 1e-10


def test_esr_criterion_runs_for_shape_ci():
    mean_a = np.array([0.0, 5.0, 1.5])
    cov_a = np.diag([0.2, 0.5, 0.2])
    mean_b = np.array([0.2, 4.8, 1.8])
    cov_b = np.diag([0.1, 0.8, 0.3])

    fused = fuse_gaussians_unknown_correlation(
        mean_a,
        cov_a,
        mean_b,
        cov_b,
        method="ci",
        criterion="esr_trace",
        grid_size=9,
    )

    assert 0.0 <= fused.omega <= 1.0
    assert np.all(np.linalg.eigvalsh(fused.covariance) > 0.0)



def test_covariance_union_inflation_covers_separated_inputs():
    mean_f = np.array([5.0, 0.0])
    cov_f = np.eye(2)
    mean_a = np.array([0.0, 0.0])
    mean_b = np.array([10.0, 0.0])
    cov = np.eye(2)

    cov_u = covariance_union_inflate(mean_f, cov_f, [(mean_a, cov), (mean_b, cov)])

    assert np.all(np.linalg.eigvalsh(cov_u) > 0.0)
    assert cov_u[0, 0] > 25.0
    for mean in (mean_a, mean_b):
        diff = mean - mean_f
        assert float(diff.T @ np.linalg.inv(cov_u) @ diff) < 1.0


def test_ci_cu_uses_union_fallback_when_gate_is_tight():
    mean_a = np.array([0.0, 0.0])
    mean_b = np.array([10.0, 0.0])
    cov = np.eye(2)

    ci = fuse_gaussians_unknown_correlation(mean_a, cov, mean_b, cov, method="ci", fixed_omega=0.5)
    ci_cu = fuse_gaussians_unknown_correlation(
        mean_a,
        cov,
        mean_b,
        cov,
        method="ci_cu",
        fixed_omega=0.5,
        cu_gate_threshold=0.0,
    )

    np.testing.assert_allclose(ci_cu.mean, ci.mean, atol=1e-12)
    assert np.linalg.det(ci_cu.covariance) > np.linalg.det(ci.covariance)


def test_ci_cu_does_not_inflate_identical_inputs():
    mean = np.array([0.3, 4.0, 1.5])
    cov = np.diag([0.2, 0.5, 0.4])

    ci_cu = fuse_gaussians_unknown_correlation(mean, cov, mean, cov, method="ci_cu", fixed_omega=0.5)

    np.testing.assert_allclose(ci_cu.mean, mean, atol=1e-12)
    np.testing.assert_allclose(ci_cu.covariance, cov, atol=1e-12)
