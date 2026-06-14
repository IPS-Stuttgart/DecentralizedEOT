"""Evaluate centralized MEM-EKF* ANEES with all sensor measurements every scan."""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal as mvn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--time-steps", type=int, default=15)
    parser.add_argument("--scenario", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="plots_centralized_mem_anees")
    return parser.parse_args()


def _patch_constants(args: argparse.Namespace):
    import constants as C

    C.SCENARIO_ID = args.scenario
    C.SIGMA_OR = np.sqrt(0.01 * np.pi) if args.scenario == 0 else np.sqrt(0.2 * np.pi)
    C.RM_FORGET = 6.0 if args.scenario == 0 else 3.0
    C.RUNS = args.runs
    C.TIME_STEPS = args.time_steps
    C.DIRECT = False
    C.NO_KIN = False
    C.LOAD_DATA = False
    return C


def _copy_measurements(meas):
    return [cur.copy() for cur in meas]


def _write_csv(path, kin, shape, joint):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_step",
            "kin_anees",
            "kin_nanees",
            "shape_anees",
            "shape_nanees",
            "joint_anees",
            "joint_nanees",
        ])
        for step_id in range(len(kin)):
            writer.writerow([
                step_id,
                kin[step_id],
                kin[step_id] / 4.0,
                shape[step_id],
                shape[step_id] / 3.0,
                joint[step_id],
                joint[step_id] / 7.0,
            ])


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    C = _patch_constants(args)

    from configs import get_configs
    from Data.simulation import simulate_data
    from ErrorAndPlotting.error import (
        esr_shape_nees_from_mixture,
        joint_kinematic_esr_nees_from_mixture,
        kinematic_nees,
    )
    from Filters.memekfstar import MemEkfStarTracker

    datasets = []
    for _ in range(args.runs):
        init_state = mvn(C.INIT_STATE, C.INIT_STATE_COV)
        init_state[C.L] = np.max([init_state[C.L], C.AX_MIN])
        init_state[C.W] = np.max([init_state[C.W], C.AX_MIN])
        datasets.append([
            (gt.copy(), _copy_measurements(meas))
            for gt, meas in simulate_data(init_state, C.MEAS_COV)
        ])

    _, ax = plt.subplots(1, 1)
    config_memekfstar1, *_ = get_configs(C.INIT_STATE, ax)
    config_memekfstar1 = dict(config_memekfstar1)
    config_memekfstar1.update({"name": "Centralized MEM-EKF*", "color": "black"})
    tracker = MemEkfStarTracker(C.MEM_H, C.MEM_KIN_DYM, C.MEM_SHAPE_DYM, C.MEAS_COV1, **config_memekfstar1)

    anees_kin = np.zeros(args.time_steps)
    anees_shape = np.zeros(args.time_steps)
    anees_joint = np.zeros(args.time_steps)

    for run_id, run_data in enumerate(datasets):
        print(f"Starting run {run_id + 1} of {args.runs}")
        for step_id, (gt, meas) in enumerate(run_data):
            td = 0.0 if step_id == 0 else C.TD
            if td > 0.0:
                tracker.predict(td)

            # Centralized processing: both sensors' measurements at this scan,
            # each with its own sensor covariance.
            tracker.correct(meas[0].copy(), C.MEAS_COV1.copy())
            tracker.correct(meas[1].copy(), C.MEAS_COV2.copy())

            est, est_cov, _ = tracker.get_est()
            kin_cov = est_cov[:4, :4]
            shape_mean = est[None, [C.AL, C.L, C.W]]
            shape_cov = est_cov[None, 4:, 4:]
            weights = np.ones(1)

            anees_kin[step_id] += kinematic_nees(est, gt, kin_cov)
            anees_shape[step_id] += esr_shape_nees_from_mixture(
                est[[C.AL, C.L, C.W]],
                gt[[C.AL, C.L, C.W]],
                shape_mean,
                shape_cov,
                weights,
            )
            anees_joint[step_id] += joint_kinematic_esr_nees_from_mixture(
                est,
                gt,
                kin_cov,
                shape_mean,
                shape_cov,
                weights,
            )

        tracker.reset(C.INIT_STATE, C.INIT_STATE_COV)

    anees_kin /= args.runs
    anees_shape /= args.runs
    anees_joint /= args.runs

    np.savez(
        os.path.join(args.out_dir, "centralized_mem_anees.npz"),
        kin=anees_kin,
        shape=anees_shape,
        joint=anees_joint,
    )
    _write_csv(os.path.join(args.out_dir, "centralized_mem_anees.csv"), anees_kin, anees_shape, anees_joint)

    print("\nFinal-step centralized MEM-EKF* ANEES")
    print("=====================================")
    print("Raw targets: kinematic=4, ESR-shape=3, joint ESR=7")
    print("Normalized target: 1.0")
    print(f"kin={anees_kin[-1]:.6f} ({anees_kin[-1] / 4.0:.6f})")
    print(f"shape={anees_shape[-1]:.6f} ({anees_shape[-1] / 3.0:.6f})")
    print(f"joint={anees_joint[-1]:.6f} ({anees_joint[-1] / 7.0:.6f})")
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
