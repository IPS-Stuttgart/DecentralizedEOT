"""Confirm the heavy-tail mechanism: is ANEES dominated by covariance-collapse runs?"""
from __future__ import annotations
import numpy as np
from numpy.random import multivariate_normal as mvn


def main():
    import os
    runs = int(os.environ.get("DIAG_RUNS", 1000))
    time_steps = int(os.environ.get("DIAG_STEPS", 15))
    seed = 7
    np.random.seed(seed)
    import constants as C
    C.SCENARIO_ID = 1
    C.SIGMA_OR = np.sqrt(0.2 * np.pi)
    C.DIRECT = False; C.NO_KIN = False; C.LOAD_DATA = False
    C.RUNS = runs; C.TIME_STEPS = time_steps

    from Data.simulation import simulate_data
    from Filters.memekfstar import MemEkfStarTracker
    from ErrorAndPlotting.error import shape_to_esr, shape_to_esr_jacobian

    datasets = []
    for _ in range(runs):
        s0 = mvn(C.INIT_STATE, C.INIT_STATE_COV)
        s0[C.L] = max(s0[C.L], C.AX_MIN); s0[C.W] = max(s0[C.W], C.AX_MIN)
        datasets.append([(gt.copy(), [m.copy() for m in meas])
                         for gt, meas in simulate_data(s0, C.MEAS_COV)])

    cfg = {'init_state': C.INIT_STATE, 'init_cov': C.INIT_STATE_COV, 'init_rate': C.INIT_RATE,
           'time_steps': time_steps, 'ax': None, 'name': 'd', 'color': 'k', 'exist_prob': 1.0,
           'Q': np.array([C.SIGMA_V1, C.SIGMA_V2]), 'SH': np.array([C.SIGMA_OR, 0.001, 0.001]),
           'al_approx': False, 'mmgw': False}
    tr = MemEkfStarTracker(C.MEM_H, C.MEM_KIN_DYM, C.MEM_SHAPE_DYM, C.MEAS_COV1, **cfg)

    nees = np.zeros(runs)
    detP = np.zeros(runs)
    std_a = np.zeros(runs); err_a = np.zeros(runs)
    for r, run_data in enumerate(datasets):
        for step_id, (gt, meas) in enumerate(run_data):
            td = 0.0 if step_id == 0 else C.TD
            if td > 0.0:
                tr.predict(td)
            tr.correct(meas[0].copy(), C.MEAS_COV1.copy())
            tr.correct(meas[1].copy(), C.MEAS_COV2.copy())
        est, est_cov, _ = tr.get_est()
        P = est_cov[4:, 4:]
        e = shape_to_esr(est[[C.AL, C.L, C.W]]) - shape_to_esr(gt[[C.AL, C.L, C.W]])
        J = shape_to_esr_jacobian(est[[C.AL, C.L, C.W]])
        Pe = J @ P @ J.T
        nees[r] = e @ np.linalg.pinv(Pe) @ e
        detP[r] = np.linalg.det(Pe)
        std_a[r] = np.sqrt(max(est_cov[C.L, C.L], 0))   # reported semi-major std
        err_a[r] = abs(est[C.L] - gt[C.L])
        tr.reset(C.INIT_STATE, C.INIT_STATE_COV)

    nn = nees / 3.0
    print("Per-run normalized ESR NEES distribution:")
    for p in [50, 75, 90, 95, 99]:
        print(f"   {p:2d}th pct: {np.percentile(nn, p):10.3f}")
    print(f"   mean    : {nn.mean():10.3f}   (this is the reported ANEES)")
    print(f"   max     : {nn.max():10.3f}")
    order = np.argsort(nn)
    k1 = max(1, int(runs * 0.01))
    k5 = max(1, int(runs * 0.05))
    print(f"\n   share of total NEES mass from worst  1% of runs: {nn[order[-k1:]].sum()/nn.sum():6.1%}")
    print(f"   share of total NEES mass from worst  5% of runs: {nn[order[-k5:]].sum()/nn.sum():6.1%}")
    print(f"   ANEES if worst 1% dropped: {nn[order[:-k1]].mean():8.3f}")
    print(f"   ANEES if worst 5% dropped: {nn[order[:-k5]].mean():8.3f}")

    lo, hi = nn < np.median(nn), nn >= np.percentile(nn, 99)
    print(f"\n   reported semi-major std:  median-NEES runs={std_a[lo].mean():.4f}   "
          f"worst-1% runs={std_a[hi].mean():.4f}")
    print(f"   |semi-major error|     :  median-NEES runs={err_a[lo].mean():.4f}   "
          f"worst-1% runs={err_a[hi].mean():.4f}")
    print(f"   -> worst runs have {'SMALL cov (collapse)' if std_a[hi].mean()<std_a[lo].mean() else 'large error'}")


if __name__ == "__main__":
    main()
