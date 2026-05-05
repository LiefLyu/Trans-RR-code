"""Reviewer 2 Comment 3 -- ridge penalty grid-width sensitivity (Tab B4).

Compares the §4.3 main comparison (which uses the default TAU_GRID) with the
same comparison run on a wider grid:
    default: TAU_GRID       = np.logspace(-2, 2, 9, base=3)   (9 points, [1/9, 9])
    wide:    WIDE_TAU_GRID  = np.logspace(-3, 3, 13, base=3)  (13 points, [1/27, 27])
TAU_GRID is a strict subset of WIDE_TAU_GRID.

This driver runs only the WIDE grid; the default-grid column in Tab B4 is
read from the main JSON files (2{gaussian,cauchy,mix}_cv_results_p400_simu500
.json), which are byte-identical for the four ridge methods given that ridge
is computed via deterministic L-BFGS-B and KFold(seed=1) regardless of whether
the calling pipeline includes Lasso. Skipping the default-grid sub-run saves
about 21 cells x M=500 of redundant computation.

Setup: 3 cases × 7 h × 1 grid (wide) × 4 ridge methods × M reps.
Methods: Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR.

Output: res/2sensitivity_tau_p400_simu{M}.json with the per-cell schema:
    {
        "case", "dd", "grid_label" = "wide",
        "mean_err":   [Single, Trans, Ada, Pooled],
        "std_err":    [Single, Trans, Ada, Pooled],
        "errs":       per-method per-rep arrays,
        "tau_st", "tau_src", "tau_tgt", "tau_pool", "theta_star": per-rep arrays.
    }
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from transrr_lib._grids import TAU_GRID, WIDE_TAU_GRID
from pipelines import estimate_regression_models_noLASSO


# Same data generators as run_*_main.py
def make_data_gaussian(seed, p, n, nn, beta_0, w_0, sigma, sigma1):
    np.random.seed(seed)
    X = np.random.normal(size=(n, p))
    y = X @ beta_0 + np.random.normal(0, sigma, n)
    X1 = np.random.normal(size=(nn, p))
    y1 = X1 @ w_0 + np.random.normal(0, sigma1, nn)
    return X, y, X1, y1


def make_data_cauchy(seed, p, n, nn, beta_0, w_0, sigma, sigma1):
    np.random.seed(seed)
    lam = np.random.uniform(0, np.sqrt(3), n)
    X = np.random.normal(size=(n, p)) * lam[:, np.newaxis]
    y = X @ beta_0 + np.random.standard_cauchy(size=n) * sigma
    lam1 = np.random.uniform(0, np.sqrt(3), nn)
    X1 = np.random.normal(size=(nn, p)) * lam1[:, np.newaxis]
    y1 = X1 @ w_0 + np.random.standard_cauchy(size=nn) * sigma1
    return X, y, X1, y1


def make_data_mix(seed, p, n, nn, beta_0, w_0, sigma, sigma1):
    np.random.seed(seed)
    lam = np.random.uniform(0, np.sqrt(3), n // 2)
    X11 = np.random.normal(size=(n // 2, p)) * lam[:, np.newaxis]
    Y11 = X11 @ beta_0 + np.random.standard_cauchy(size=n // 2)
    X12 = np.random.normal(size=(n // 2, p))
    Y12 = X12 @ beta_0 + np.random.normal(0, sigma, n // 2)
    X = np.vstack((X11, X12)); Y = np.concatenate((Y11, Y12))
    lam1 = np.random.uniform(0, np.sqrt(3), nn // 2)
    X21 = np.random.normal(size=(nn // 2, p)) * lam1[:, np.newaxis]
    Y21 = X21 @ w_0 + np.random.standard_cauchy(size=nn // 2) * sigma1
    X22 = np.random.normal(size=(nn // 2, p))
    Y22 = X22 @ w_0 + np.random.normal(0, sigma1, nn // 2)
    X1 = np.vstack((X21, X22)); y1 = np.concatenate((Y21, Y22))
    return X, Y, X1, y1


DATA_GENERATORS = {
    "gaussian": make_data_gaussian,
    "cauchy":   make_data_cauchy,
    "mix":      make_data_mix,
}


def worker(seed, case, p, n, nn, beta_0, w_0, sigma, sigma1, tau_range):
    X, y, X1, y1 = DATA_GENERATORS[case](seed, p, n, nn, beta_0, w_0, sigma, sigma1)
    res = estimate_regression_models_noLASSO(
        X, y, X1, y1,
        delta_param=1.35, eta_param=0.1,
        tau_range=tau_range,
        criterion='mae', loss='smoothed_huber',
    )
    keys = ["single_robust_ridge", "transfer_robust_ridge", "adaptive", "pooled_robust_ridge"]
    errnorms = tuple(np.sum((res[k]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2) for k in keys)
    taus = (
        float(res["single_robust_ridge"]["optimal_tau"]),
        float(res["transfer_robust_ridge"]["optimal_tau_source"]),
        float(res["transfer_robust_ridge"]["optimal_tau_target_diff"]),
        float(res["pooled_robust_ridge"]["optimal_tau"]),
    )
    theta_star = float(res["adaptive"]["theta_star"])
    return errnorms + taus + (theta_star,)  # 4 errs + 4 taus + 1 theta


def run_one_cell(case, dd, grid_label, p, n, K, n_jobs, tau_range):
    nn = 2 * n
    sigma, sigma1 = 1, 2

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0

    desc = f"{case} h={dd:.3f} grid={grid_label}"
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            seed, case, p, n, nn, beta_0, w_0, sigma, sigma1, tau_range
        )
        for seed in tqdm(range(K), desc=desc)
    )
    arr = np.array(results)
    errnorms = arr[:, :4]
    tau_values = arr[:, 4:8]
    theta_values = arr[:, 8]
    return {
        "case":       case,
        "dd":         float(dd),
        "grid_label": grid_label,
        "tau_grid":   [float(x) for x in tau_range],
        "mean_err":   errnorms.mean(axis=0).tolist(),
        "std_err":    errnorms.std(axis=0).tolist(),
        "errs": {
            "Single RR":     errnorms[:, 0].tolist(),
            "Trans RR":      errnorms[:, 1].tolist(),
            "Trans-RR-Ada":  errnorms[:, 2].tolist(),
            "Pooled RR":     errnorms[:, 3].tolist(),
        },
        "tau_st":   tau_values[:, 0].tolist(),
        "tau_src":  tau_values[:, 1].tolist(),
        "tau_tgt":  tau_values[:, 2].tolist(),
        "tau_pool": tau_values[:, 3].tolist(),
        "theta_star": theta_values.tolist(),
    }


def main():
    p = 400
    n = 400
    K = 500
    n_jobs = 11

    cases = ["gaussian", "cauchy", "mix"]
    dd_values = list(np.power(np.e, np.arange(-2.0, 1.5, 0.5)))   # 7 h values
    # Skip "default" grid because it duplicates the main experiment.
    # The default column of Tab B4 is read from main JSONs by the renderer.
    grids = [
        ("wide", WIDE_TAU_GRID),
    ]

    print("=== Ridge-grid-width sensitivity (Tab B4) ===")
    print(f"p={p}, n={n}, K={K}, n_jobs={n_jobs}")
    print(f"default grid ({len(TAU_GRID)} pts) [main JSON, not re-run here]:")
    print(f"  {[f'{x:.3g}' for x in TAU_GRID]}")
    print(f"wide grid    ({len(WIDE_TAU_GRID)} pts) [run here]:")
    print(f"  {[f'{x:.3g}' for x in WIDE_TAU_GRID]}")
    print(f"cases: {cases}")
    print(f"h values: {[f'{x:.3f}' for x in dd_values]}")
    print(f"-> {len(cases) * len(dd_values) * len(grids)} cells x M={K}")
    print()

    cells = []
    t0 = time.time()
    for case in cases:
        for dd in dd_values:
            for grid_label, tau_range in grids:
                cell = run_one_cell(case, dd, grid_label, p, n, K, n_jobs, tau_range)
                cells.append(cell)
                print(f"  {case} h={dd:.3f} {grid_label:7}  mean = {[f'{x:.4f}' for x in cell['mean_err']]}")

    payload = {
        "p":         p,
        "n":         n,
        "K":         K,
        "cases":     cases,
        "dd_values": [float(x) for x in dd_values],
        "default_grid": [float(x) for x in TAU_GRID],
        "wide_grid":    [float(x) for x in WIDE_TAU_GRID],
        "cells":     cells,
        "method_names": ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"],
        "wall_time_minutes": (time.time() - t0) / 60.0,
    }
    Path("res").mkdir(exist_ok=True)
    filename = f"res/2sensitivity_tau_p{p}_simu{K}.json"
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved: {filename}")
    print(f"Total wall time: {payload['wall_time_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
