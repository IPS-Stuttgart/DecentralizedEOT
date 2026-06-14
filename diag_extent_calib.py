"""Diagnostic: where does the centralized MEM-EKF* extent overconfidence come from?

Decomposes the final-step shape error into per-component bias vs variance, in both
raw (alpha,a,b) and ESR space, and sweeps the axis process noise on identical data.
"""
from __future__ import annotations

import numpy as np
from numpy.random import multivariate_normal as mvn


def patch_constants():
    import constants as C
    C.SCENARIO_ID = 1
    C.SIGMA_OR = np.sqrt(0.2 * np.pi)
    C.RM_FORGET = 3.0
    C.DIRECT = False
    C.NO_KIN = False
    C.LOAD_DATA = False
    return C


def wrap_pm_halfpi(d):
    # ellipse orientation is pi-periodic
    return (d + np.pi / 2.0) % np.pi - np.pi / 2.0


def main():
    import os
    runs = int(os.environ.get("DIAG_RUNS", 1000))
    time_steps = int(os.environ.get("DIAG_STEPS", 15))
    seed = 7
    np.random.seed(seed)

    C = patch_constants()
    C.RUNS = runs
    C.TIME_STEPS = time_steps

    from Data.simulation import simulate_data
    from Filters.memekfstar import MemEkfStarTracker
    from ErrorAndPlotting.error import shape_to_esr, shape_to_esr_jacobian

    # Build datasets ONCE so every SH config sees identical data.
    datasets = []
    for _ in range(runs):
        init_state = mvn(C.INIT_STATE, C.INIT_STATE_COV)
        init_state[C.L] = np.max([init_state[C.L], C.AX_MIN])
        init_state[C.W] = np.max([init_state[C.W], C.AX_MIN])
        datasets.append([(gt.copy(), [m.copy() for m in meas])
                         for gt, meas in simulate_data(init_state, C.MEAS_COV)])

    def base_config(sh):
        return {
            'init_state': C.INIT_STATE, 'init_cov': C.INIT_STATE_COV, 'init_rate': C.INIT_RATE,
            'time_steps': time_steps, 'ax': None, 'name': 'diag', 'color': 'black',
            'exist_prob': 1.0,
            'Q': np.array([C.SIGMA_V1, C.SIGMA_V2]),
            'SH': np.array([C.SIGMA_OR, sh, sh]),
            'al_approx': False, 'mmgw': False,
        }

    def run_filter(sh):
        cfg = base_config(sh)
        tr = MemEkfStarTracker(C.MEM_H, C.MEM_KIN_DYM, C.MEM_SHAPE_DYM, C.MEAS_COV1, **cfg)
        raw_err = np.zeros((runs, 3))     # alpha(wrapped), a, b
        esr_err = np.zeros((runs, 3))
        raw_cov = np.zeros((runs, 3, 3))
        esr_cov = np.zeros((runs, 3, 3))
        for r, run_data in enumerate(datasets):
            for step_id, (gt, meas) in enumerate(run_data):
                td = 0.0 if step_id == 0 else C.TD
                if td > 0.0:
                    tr.predict(td)
                tr.correct(meas[0].copy(), C.MEAS_COV1.copy())
                tr.correct(meas[1].copy(), C.MEAS_COV2.copy())
            est, est_cov, _ = tr.get_est()
            sh_est = est[[C.AL, C.L, C.W]]
            sh_gt = gt[[C.AL, C.L, C.W]]
            P = est_cov[4:, 4:]
            d = sh_est - sh_gt
            d[0] = wrap_pm_halfpi(d[0])
            raw_err[r] = d
            raw_cov[r] = P
            e = shape_to_esr(sh_est) - shape_to_esr(sh_gt)
            J = shape_to_esr_jacobian(sh_est)
            esr_err[r] = e
            esr_cov[r] = J @ P @ J.T
            tr.reset(C.INIT_STATE, C.INIT_STATE_COV)
        return raw_err, esr_err, raw_cov, esr_cov

    def decompose(err, cov, names):
        Pbar = cov.mean(axis=0)
        bias = err.mean(axis=0)
        S = np.cov(err, rowvar=False, bias=True)  # scatter about actual mean
        Pinv = np.linalg.pinv(Pbar)
        var_term = np.trace(Pinv @ S)
        bias_term = bias @ Pinv @ bias
        total = var_term + bias_term
        print(f"   {'comp':>6} {'RMSE':>9} {'rep.std':>9} {'bias':>9} "
              f"{'var ratio':>10} {'bias^2/var':>11}")
        for i, nm in enumerate(names):
            rmse = np.sqrt((err[:, i] ** 2).mean())
            repstd = np.sqrt(Pbar[i, i])
            var_ratio = S[i, i] / Pbar[i, i]
            bias2_ratio = bias[i] ** 2 / Pbar[i, i]
            print(f"   {nm:>6} {rmse:9.4f} {repstd:9.4f} {bias[i]:9.4f} "
                  f"{var_ratio:10.2f} {bias2_ratio:11.2f}")
        print(f"   NEES total={total:7.2f}  (var term={var_term:6.2f}, "
              f"bias term={bias_term:6.2f})  normalized={total/3.0:6.2f}")
        return total / 3.0

    print("=" * 78)
    print("BASELINE  SH=[SIGMA_OR, 0.001, 0.001]   (default)")
    print("=" * 78)
    raw_err, esr_err, raw_cov, esr_cov = run_filter(0.001)
    print(" RAW (alpha,a,b):")
    decompose(raw_err, raw_cov, ["alpha", "a", "b"])
    print(" ESR (s11,s12,s22):")
    decompose(esr_err, esr_cov, ["s11", "s12", "s22"])

    print()
    print("=" * 78)
    print("AXIS PROCESS-NOISE SWEEP  SH=[SIGMA_OR, s, s]  -> ESR shape nanees")
    print("=" * 78)
    print(f"   {'s (axis SH std)':>16} {'ESR nanees':>12}")
    for s in [0.001, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        _, esr_e, _, esr_c = run_filter(s)
        Pinv = np.linalg.pinv(esr_c.mean(axis=0))
        # per-run NEES averaged (matches the repo's ANEES definition)
        nees = np.mean([esr_e[r] @ np.linalg.pinv(esr_c[r]) @ esr_e[r] for r in range(runs)])
        print(f"   {s:16.3f} {nees/3.0:12.3f}")


if __name__ == "__main__":
    main()
