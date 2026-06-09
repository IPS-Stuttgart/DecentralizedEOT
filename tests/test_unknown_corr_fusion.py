"""Small tests for the RED unknown-correlation add-on.

Run from the repository root after copying the patch files:

    python -m pytest tests/test_unknown_corr_fusion.py
"""

import numpy as np

from Filters.filtersupport import turn_mult
from Filters.fusioncenter_unknowncorr import (
    align_shape_mean_to_reference,
    fuse_gaussians_unknown_correlation,
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
