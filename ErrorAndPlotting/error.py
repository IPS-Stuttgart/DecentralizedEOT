import numpy as np
from scipy.linalg import sqrtm
from shapely.geometry import Polygon

from Filters.filtersupport import rot
from constants import *


def to_matrix(alpha, ax_l, ax_w, sr):
    """
    Turn ellipse parameters into a matrix or square root matrix depending on sr parameter
    :param alpha:   Orientation of the ellipse
    :param ax_l:    Semi-axis length of the ellipse
    :param ax_w:    Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        Shape or square root matrix depending of sr
    """
    p = 1 if sr else 2
    rot_m = rot(alpha)
    return np.dot(np.dot(rot_m, np.diag([ax_l, ax_w]) ** p), rot_m.T)


def gw_error(x, gt):
    """
    Calculates the squared Gaussian Wasserstein metric for two ellipses.
    :param x:   first ellipse, must be parameterized with center, orientation, and semi-axes
    :param gt:  second ellipse, must be parameterized with center, orientation, and semi-axes
    :return:    the squared Gaussian Wasserstein distance between the two ellipses
    """
    gt_sigma = to_matrix(gt[AL], gt[L], gt[W], False)
    gt_sigma += gt_sigma.T
    gt_sigma /= 2.0

    track_sigma = to_matrix(x[AL], x[L], x[W], False)
    track_sigma += track_sigma.T
    track_sigma /= 2.0

    error = np.linalg.norm(gt[[X1, X2]] - x[[X1, X2]]) ** 2 \
            + np.trace(gt_sigma + track_sigma - 2 * sqrtm(np.einsum('ab, bc, cd -> ad', sqrtm(gt_sigma), track_sigma,
                                                                    sqrtm(gt_sigma))))

    return error


def iou_error(x, gt):
    """
    Calculates intersection-over-union between the two ellipses.
    :param x:   first ellipse, must be parameterized with center, orientation, and semi-axes
    :param gt:  second ellipse, must be parameterized with center, orientation, and semi-axes
    :return:    the iou value between 0 and 1
    """
    # get points on ellipses
    theta = np.linspace(0.0, 2.0*np.pi, 100)
    x_points = x[[X1, X2], None] + rot(x[AL]) @ np.diag([x[L], x[W]]) @ np.array([np.cos(theta), np.sin(theta)])
    gt_points = gt[[X1, X2], None] + rot(gt[AL]) @ np.diag([gt[L], gt[W]]) @ np.array([np.cos(theta), np.sin(theta)])

    # create polygon
    x_pol = Polygon(x_points.T)
    gt_pol = Polygon(gt_points.T)

    # calculate IoU
    intersec = gt_pol.intersection(x_pol).area
    return intersec / (gt_pol.area + x_pol.area - intersec)


def kinematic_nees(x, gt, kin_cov):
    """
    Calculates the normalized estimation error squared for the 4D kinematic state.
    :param x:        state estimate, must contain [x1, x2, v1, v2]
    :param gt:       ground truth, must contain [x1, x2, v1, v2]
    :param kin_cov:  4x4 covariance of the kinematic state estimate
    :return:         NEES value for the kinematic state
    """
    kin_cov = np.asarray(kin_cov, dtype=float)
    kin_cov = 0.5 * (kin_cov + kin_cov.T)
    err = x[[X1, X2, V1, V2]] - gt[[X1, X2, V1, V2]]
    return float(err.T @ np.linalg.pinv(kin_cov) @ err)


def shape_to_esr(shape):
    """
    Converts ellipse parameters [alpha, l, w] into ESR coordinates [s11, s12, s22].
    The ESR matrix is the symmetric square root of the extent matrix.
    """
    alpha, ax_l, ax_w = np.asarray(shape, dtype=float)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    cos_sq = cos_a ** 2
    sin_sq = sin_a ** 2
    sin_cos = sin_a * cos_a

    return np.array([
        ax_l * cos_sq + ax_w * sin_sq,
        (ax_l - ax_w) * sin_cos,
        ax_l * sin_sq + ax_w * cos_sq,
    ])


def shape_to_esr_jacobian(shape):
    """
    Jacobian of ``shape_to_esr`` with respect to [alpha, l, w].
    """
    alpha, ax_l, ax_w = np.asarray(shape, dtype=float)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    cos_sq = cos_a ** 2
    sin_sq = sin_a ** 2
    sin_cos = sin_a * cos_a
    cos_2a = cos_sq - sin_sq

    return np.array([
        [2.0 * (ax_w - ax_l) * sin_cos, cos_sq, sin_sq],
        [(ax_l - ax_w) * cos_2a, sin_cos, -sin_cos],
        [2.0 * (ax_l - ax_w) * sin_cos, sin_sq, cos_sq],
    ])


def shape_cov_to_esr(shape, shape_cov):
    """
    First-order covariance transform from [alpha, l, w] into ESR coordinates.
    """
    shape_cov = np.asarray(shape_cov, dtype=float)
    shape_cov = 0.5 * (shape_cov + shape_cov.T)
    jac = shape_to_esr_jacobian(shape)
    esr_cov = jac @ shape_cov @ jac.T
    return 0.5 * (esr_cov + esr_cov.T)


def esr_mixture_moments(shape_means, shape_covs, weights):
    """
    Approximate ESR mean/covariance for a Gaussian mixture in [alpha, l, w].
    Each component covariance is transformed by a local first-order ESR Jacobian.
    """
    shape_means = np.atleast_2d(np.asarray(shape_means, dtype=float))
    shape_covs = np.asarray(shape_covs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    esr_means = np.array([shape_to_esr(mean) for mean in shape_means])
    esr_covs = np.array([
        shape_cov_to_esr(mean, cov)
        for mean, cov in zip(shape_means, shape_covs)
    ])
    esr_mean = np.sum(weights[:, None] * esr_means, axis=0)
    centered = esr_means - esr_mean
    spread = np.sum(weights[:, None, None] * np.einsum('xa,xb->xab', centered, centered), axis=0)
    esr_cov = np.sum(weights[:, None, None] * esr_covs, axis=0) + spread
    return esr_mean, 0.5 * (esr_cov + esr_cov.T)


def esr_shape_nees_from_mixture(shape_est, shape_gt, shape_means, shape_covs, weights):
    """
    NEES for shape in ESR coordinates using a raw-parameter Gaussian mixture covariance.
    """
    _, esr_cov = esr_mixture_moments(shape_means, shape_covs, weights)
    err = shape_to_esr(shape_est) - shape_to_esr(shape_gt)
    return float(err.T @ np.linalg.pinv(esr_cov) @ err)


def joint_kinematic_esr_nees_from_mixture(est, gt, kin_cov, shape_means, shape_covs, weights):
    """
    Joint NEES for [x, y, vx, vy, s11, s12, s22] with block-diagonal covariance.
    """
    _, esr_cov = esr_mixture_moments(shape_means, shape_covs, weights)
    kin_cov = np.asarray(kin_cov, dtype=float)
    kin_cov = 0.5 * (kin_cov + kin_cov.T)
    joint_cov = np.zeros((7, 7))
    joint_cov[:4, :4] = kin_cov
    joint_cov[4:, 4:] = esr_cov

    err = np.hstack([
        est[[X1, X2, V1, V2]] - gt[[X1, X2, V1, V2]],
        shape_to_esr(est[[AL, L, W]]) - shape_to_esr(gt[[AL, L, W]]),
    ])
    return float(err.T @ np.linalg.pinv(joint_cov) @ err)
