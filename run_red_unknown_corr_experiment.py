"""Run RED fusion under known and unknown cross-correlation assumptions.

Copy this file and ``Filters/fusioncenter_unknowncorr.py`` into the root of
Fusion-Goettingen/Fusion_2022_Thormann_RED-IF, then run for a quick smoke test:

    python run_red_unknown_corr_experiment.py --runs 20 --time-steps 15 --seed 7

For a slower paper-style run:

    python run_red_unknown_corr_experiment.py --runs 1000 --time-steps 15 --seed 7 --include-ici

The script compares:
  * RED normal fusion        : original RED update, ignores track correlation.
  * RED information fusion   : original RED-IF update, known common information.
  * RED-CI unknown corr.     : new conservative unknown-correlation baseline.
  * RED-CI-CU unknown corr.  : CI with covariance-union safety inflation for inconsistent tracks.
  * RED-ICI unknown corr.    : optional sharper experimental baseline.

RED-IF should remain strong when its known-common-information assumptions are
true.  RED-CI/RED-ICI are intended for the case where that information is not
available, so the expected behavior is conservative rather than always lower
GW error.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal as mvn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100, help="Monte Carlo runs. Use 1000 for paper-like sweeps.")
    parser.add_argument("--time-steps", type=int, default=15, help="Number of scan steps per run.")
    parser.add_argument("--scenario", type=int, choices=[0, 1], default=None,
                        help="0: low orientation process noise; 1: high orientation process noise. "
                             "If omitted, constants.py's current SCENARIO_ID is used.")
    parser.add_argument("--seed", type=int, default=None, help="Optional NumPy random seed.")
    parser.add_argument("--out-dir", default="plots_unknown_corr", help="Output directory for CSV/NPZ/PNG files.")
    parser.add_argument("--omega-grid-size", type=int, default=21,
                        help="Grid points for CI/ICI omega optimization. Increase for final experiments.")
    parser.add_argument("--fixed-omega", type=float, default=None,
                        help="Set a fixed CI/ICI omega. Useful for debugging speed, e.g. 0.5.")
    parser.add_argument("--omega-criterion", choices=["logdet", "trace", "esr_trace", "esr_logdet"], default="logdet",
                        help="Backward-compatible criterion used for both kinematics and shape if the split options are omitted.")
    parser.add_argument("--kin-omega-criterion", choices=["logdet", "trace"], default=None,
                        help="Optional separate CI/ICI omega objective for the 4-D kinematic state.")
    parser.add_argument("--shape-omega-criterion", choices=["logdet", "trace", "esr_trace", "esr_logdet"], default=None,
                        help="Optional separate CI/ICI omega objective for the 3-D shape state. "
                             "The esr_* variants optimize in RED square-root shape space.")
    parser.add_argument("--component-weight-mode",
                        choices=["likelihood", "esr_likelihood", "chernoff", "prior", "uniform"],
                        default="likelihood",
                        help="How RED component weights are assigned after CI/ICI component fusion. "
                             "chernoff gives GCI-style mixture weights for CI; esr_likelihood uses square-root-space compatibility.")
    parser.add_argument("--component-pairing-mode", choices=["all", "best", "gated"], default="all",
                        help="RED component-pair selection before mixture reduction.")
    parser.add_argument("--component-gate-log-weight", type=float, default=12.0,
                        help="For --component-pairing-mode gated, keep pairs within this log-score margin of the best pair per prior component.")
    parser.add_argument("--compatibility-scale", type=float, default=1.0,
                        help="Inflation factor in the compatibility likelihood used only for RED component weights.")
    parser.add_argument("--cu-gate-threshold", type=float, default=None,
                        help="Optional NIS threshold for CI-CU covariance-union fallback. If omitted, a loose chi-square-like default is used.")
    parser.add_argument("--cu-inflation-margin", type=float, default=1e-6,
                        help="Extra multiplicative covariance inflation used by covariance-union safety variants.")
    parser.add_argument("--cu-esr-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="Use ESR/square-root shape-space compatibility for shape CI-CU gates.")
    parser.add_argument("--estimate-samples", type=int, default=1000,
                        help="Particles for MMGW/ESR point estimate. Lower for faster smoke tests.")
    parser.add_argument("--include-ci-cu", action="store_true",
                        help="Also evaluate RED-CI-CU: CI plus covariance-union safety inflation for inconsistent local tracks.")
    parser.add_argument("--include-ici", action="store_true", help="Also evaluate experimental RED-ICI.")
    parser.add_argument("--include-gci-esr", action="store_true",
                        help="Also evaluate a proposed RED-GCI/ESR preset: CI, shape esr_trace omega, Chernoff weights, gated pairs.")
    return parser.parse_args()


def _patch_constants(args: argparse.Namespace):
    """Patch constants before importing repository modules that copy constants."""

    import constants as C

    if args.scenario is not None:
        C.SCENARIO_ID = args.scenario
        C.SIGMA_OR = np.sqrt(0.01 * np.pi) if args.scenario == 0 else np.sqrt(0.2 * np.pi)
        C.RM_FORGET = 6.0 if args.scenario == 0 else 3.0
    C.RUNS = args.runs
    C.TIME_STEPS = args.time_steps
    C.LOAD_DATA = False
    return C


@dataclass
class FilterRecord:
    key: str
    label: str
    filter_obj: object


def _mean_errors(records: Iterable[FilterRecord], runs: int) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for rec in records:
        out[rec.key] = {
            "gw": rec.filter_obj._error_gw / runs,
            "iou": rec.filter_obj._error_iou / runs,
            "vel": rec.filter_obj._error_vel / runs,
        }
    return out


def _write_csv(path: str, errors: Dict[str, Dict[str, np.ndarray]]) -> None:
    keys = list(errors.keys())
    n_steps = len(next(iter(errors.values()))["gw"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time_step"]
        for key in keys:
            header.extend([f"{key}_gw", f"{key}_iou", f"{key}_vel"])
        writer.writerow(header)
        for k in range(n_steps):
            row = [k]
            for key in keys:
                row.extend([errors[key]["gw"][k], errors[key]["iou"][k], errors[key]["vel"][k]])
            writer.writerow(row)


def _plot_metric(path: str, errors: Dict[str, Dict[str, np.ndarray]], metric: str, ylabel: str) -> None:
    plt.figure()
    for key, values in errors.items():
        plt.plot(np.arange(len(values[metric])), values[metric], label=key)
    plt.xlabel("time step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = _parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    C = _patch_constants(args)

    # Import after patching constants, because several repository modules use
    # ``from constants import *`` at import time.
    from configs import get_configs
    from Data.simulation import simulate_data
    from Filters.memekfstar import MemEkfStarTracker
    from Filters.fusioncenter import FusionCenter
    from Filters.fusioncenter_unknowncorr import UnknownCorrelationFusionCenter

    _, ax = plt.subplots(1, 1)

    init_state = mvn(C.INIT_STATE, C.INIT_STATE_COV)
    init_state[C.L] = np.max([init_state[C.L], C.AX_MIN])
    init_state[C.W] = np.max([init_state[C.W], C.AX_MIN])

    (
        config_memekfstar1,
        config_memekfstar2,
        _config_fusion,
        config_fusion_red,
        _config_fusion_if,
        config_fusion_if_red,
        _config_rm1,
        _config_rm2,
        _config_fusion_rm,
        _config_direct1,
        _config_direct2,
    ) = get_configs(C.INIT_STATE, ax)

    memekfstar1 = MemEkfStarTracker(C.MEM_H, C.MEM_KIN_DYM, C.MEM_SHAPE_DYM, C.MEAS_COV1, **config_memekfstar1)
    memekfstar2 = MemEkfStarTracker(C.MEM_H, C.MEM_KIN_DYM, C.MEM_SHAPE_DYM, C.MEAS_COV2, **config_memekfstar2)

    fusion_red = FusionCenter(**config_fusion_red)
    fusion_red_if = FusionCenter(**config_fusion_if_red)

    config_red_ci = dict(config_fusion_red)
    config_red_ci.update({
        "name": "RED-CI unknown corr.",
        "color": "black",
        "unknown_corr_method": "ci",
        "omega_criterion": args.omega_criterion,
        "kin_omega_criterion": args.kin_omega_criterion or args.omega_criterion,
        "shape_omega_criterion": args.shape_omega_criterion or args.omega_criterion,
        "fixed_omega": args.fixed_omega,
        "omega_grid_size": args.omega_grid_size,
        "component_weight_mode": args.component_weight_mode,
        "component_pairing_mode": args.component_pairing_mode,
        "component_gate_log_weight": args.component_gate_log_weight,
        "compatibility_scale": args.compatibility_scale,
        "cu_gate_threshold": args.cu_gate_threshold,
        "cu_inflation_margin": args.cu_inflation_margin,
        "cu_esr_gate": args.cu_esr_gate,
        "estimate_samples": args.estimate_samples,
    })
    fusion_red_ci = UnknownCorrelationFusionCenter(**config_red_ci)

    records: List[FilterRecord] = [
        FilterRecord("red_normal", "RED normal fusion", fusion_red),
        FilterRecord("red_if", "RED information fusion", fusion_red_if),
        FilterRecord("red_ci", "RED-CI unknown corr.", fusion_red_ci),
    ]

    if args.include_ci_cu:
        config_red_ci_cu = dict(config_red_ci)
        config_red_ci_cu.update({
            "name": "RED-CI-CU unknown corr.",
            "color": "brown",
            "unknown_corr_method": "ci_cu",
        })
        fusion_red_ci_cu = UnknownCorrelationFusionCenter(**config_red_ci_cu)
        records.append(FilterRecord("red_ci_cu", "RED-CI-CU unknown corr.", fusion_red_ci_cu))

    if args.include_gci_esr:
        config_red_gci_esr = dict(config_red_ci)
        config_red_gci_esr.update({
            "name": "RED-GCI/ESR unknown corr.",
            "color": "purple",
            "unknown_corr_method": "ci",
            "kin_omega_criterion": "logdet",
            "shape_omega_criterion": "esr_trace",
            "component_weight_mode": "chernoff",
            "component_pairing_mode": "gated",
        })
        fusion_red_gci_esr = UnknownCorrelationFusionCenter(**config_red_gci_esr)
        records.append(FilterRecord("red_gci_esr", "RED-GCI/ESR unknown corr.", fusion_red_gci_esr))

    if args.include_ici:
        config_red_ici = dict(config_red_ci)
        config_red_ici.update({
            "name": "RED-ICI unknown corr.",
            "color": "gray",
            "unknown_corr_method": "ici",
        })
        fusion_red_ici = UnknownCorrelationFusionCenter(**config_red_ici)
        records.append(FilterRecord("red_ici", "RED-ICI unknown corr.", fusion_red_ici))

    for run_id in range(args.runs):
        print(f"Starting run {run_id + 1} of {args.runs}")
        simulator = simulate_data(init_state, C.MEAS_COV)

        for step_id, (gt, meas) in enumerate(simulator):
            td = 0.0 if step_id == 0 else C.TD

            memekfstar1.step(meas[0].copy(), C.MEAS_COV[0].copy(), td, step_id, gt, False)
            memekfstar2.step(meas[1].copy(), C.MEAS_COV[1].copy(), td, step_id, gt, False)

            est1, est_cov1, _ = memekfstar1.get_est()
            est2, est_cov2, _ = memekfstar2.get_est()
            estimates = np.vstack([est1, est2])
            covariances = np.stack([est_cov1, est_cov2])

            for rec in records:
                rec.filter_obj.step(estimates, covariances, td, step_id, gt, False)

        # Reset all filters to a new Monte Carlo initial condition.
        init_state = mvn(C.INIT_STATE, C.INIT_STATE_COV)
        init_state[C.L] = np.max([init_state[C.L], C.AX_MIN])
        init_state[C.W] = np.max([init_state[C.W], C.AX_MIN])

        memekfstar1.reset(C.INIT_STATE, C.INIT_STATE_COV)
        memekfstar2.reset(C.INIT_STATE, C.INIT_STATE_COV)
        for rec in records:
            rec.filter_obj.reset(C.INIT_STATE, C.INIT_STATE_COV)

    errors = _mean_errors(records, args.runs)
    np.savez(os.path.join(args.out_dir, "red_unknown_corr_errors.npz"), **{
        f"{key}_{metric}": value
        for key, vals in errors.items()
        for metric, value in vals.items()
    })
    _write_csv(os.path.join(args.out_dir, "red_unknown_corr_errors.csv"), errors)
    _plot_metric(os.path.join(args.out_dir, "gw_error.png"), errors, "gw", "mean squared GW error")
    _plot_metric(os.path.join(args.out_dir, "iou_error.png"), errors, "iou", "mean IoU")
    _plot_metric(os.path.join(args.out_dir, "vel_error.png"), errors, "vel", "mean velocity squared error")

    print("\nFinal-step summary")
    print("==================")
    for rec in records:
        key = rec.key
        print(
            f"{key:>12s}: GW={errors[key]['gw'][-1]:.6f}, "
            f"IoU={errors[key]['iou'][-1]:.6f}, Vel={errors[key]['vel'][-1]:.6f}"
        )
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
