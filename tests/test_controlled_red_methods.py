"""Smoke tests for controlled RED fusion method implementations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "tools" / "evaluation" / "run_controlled_unknown_corr_benchmark.py"

spec = importlib.util.spec_from_file_location("controlled_red_benchmark", BENCH_PATH)
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


REQUIRED_METHODS = [
    "red_if_correct",
    "oracle_correlated_red",
    "red_if_wrong",
    "parameter_ci",
    "red_ci_esr",
    "red_gci_esr",
    "red_cu",
    "red_ci_cu",
]


def test_required_method_names_are_registered() -> None:
    for method in REQUIRED_METHODS:
        assert method in bench.DEFAULT_METHODS
        assert method in bench.METHOD_LABELS


def test_required_methods_return_finite_7d_outputs() -> None:
    rng = np.random.default_rng(123)
    trial = bench.sample_trial(
        rng,
        rho=0.75,
        chart_mode="random",
        fixed_chart=None,
        axis_ratio=4.0,
    )

    for method in REQUIRED_METHODS:
        out = bench.fuse_method(method, trial, gamma_wrong=0.5, omega_grid_size=9)
        assert out.mean.shape == (bench.STATE_DIM,), method
        assert out.covariance.shape == (bench.STATE_DIM, bench.STATE_DIM), method
        assert np.all(np.isfinite(out.mean)), method
        assert np.all(np.isfinite(out.covariance)), method
        np.linalg.cholesky(bench.ensure_spd(out.covariance))


def test_controlled_benchmark_smoke(tmp_path) -> None:
    args = bench.argparse.Namespace(
        suite="all",
        trials=4,
        seed=7,
        rho_values="0.00,0.75",
        gamma_values="0.50,1.00",
        methods=",".join(REQUIRED_METHODS),
        omega_grid_size=5,
        rho_for_mismatch=0.75,
        rho_for_ambiguity=0.50,
        gamma_wrong_default=0.50,
        axis_ratio=3.0,
        out_dir=str(tmp_path),
    )
    rows = bench.run_suite(args)
    assert rows
    assert {r["method"] for r in rows} == set(REQUIRED_METHODS)
