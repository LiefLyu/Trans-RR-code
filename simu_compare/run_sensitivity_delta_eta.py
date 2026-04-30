"""Reviewer 2 Comment 3 + Reviewer 1 Minor 1: (delta, eta) sensitivity heatmap.

Verifies that the recommended (delta, eta) = (1.35, 0.1) is in a flat region of
parameter space, not a sweet spot. Case II (Cauchy errors) is used because that
is where the smoothed Huber loss matters most.

Setup: 12 cells of (delta, eta) grid x 1000 reps, fixed dd at the mid-point of
the §4.3 sweep (c_d = 0 -> dd = 1.0). Records Trans-RR errnorm and tau* for
each cell, plus a histogram of selected tau* across reps for diagnostic.

Output: res/2sensitivity_delta_eta_p400_simu1000.json
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

from pipelines import estimate_regression_models_noLASSO


def compute_errnorm_cell(i, p, n, nn, beta_0, w_0, sigma, sigma1,
                         psi_delta, psi_eta, tau_range):
    np.random.seed(i)
    lam = np.random.uniform(0, np.sqrt(3), n)
    X = np.random.normal(size=(n, p)) * lam[:, np.newaxis]
    Y = X @ beta_0 + np.random.standard_cauchy(size=n) * sigma

    lam1 = np.random.uniform(0, np.sqrt(3), nn)
    X1 = np.random.normal(size=(nn, p)) * lam1[:, np.newaxis]
    Y1 = X1 @ w_0 + np.random.standard_cauchy(size=nn) * sigma1

    res = estimate_regression_models_noLASSO(
        X, Y, X1, Y1,
        delta_param=psi_delta, eta_param=psi_eta,
        tau_range=tau_range,
    )
    err_tr = float(np.sum((res["transfer_robust_ridge"]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2))
    err_sr = float(np.sum((res["single_robust_ridge"]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2))
    err_ada = float(np.sum((res["adaptive"]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2))
    tau_sr = float(res["single_robust_ridge"]["optimal_tau"])
    theta_star = float(res["adaptive"]["theta_star"])
    return err_tr, err_sr, err_ada, tau_sr, theta_star


def main():
    p = 400
    n = 400
    nn = 800
    K = 1000
    n_jobs = 6  # realdata runs concurrent; 6+2 keeps 8 P-cores fully utilised. BLAS pinned to 1 thread by env vars above.

    sigma, sigma1 = 1, 2
    dd = 1.0  # mid-point of §4.3 sweep (c_d = 0 -> dd = e^0 = 1.0)
    tau_range = np.logspace(-2, 1, 10, base=3)

    delta_grid = [1.0, 1.35, 1.7, 2.0]
    eta_grid = [0.05, 0.1, 0.2]

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0

    print(f"=== (delta, eta) sensitivity heatmap (Case II Cauchy) ===")
    print(f"p={p}, n={n}, K={K} per cell, dd={dd}")
    print(f"delta grid: {delta_grid}")
    print(f"eta grid:   {eta_grid}")
    print()

    cells = {}
    t0 = time.time()
    for d in delta_grid:
        for e in eta_grid:
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_errnorm_cell)(
                    i, p, n, nn, beta_0, w_0, sigma, sigma1, d, e, tau_range
                )
                for i in tqdm(range(K), desc=f"delta={d:.2f} eta={e:.2f}")
            )
            arr = np.array(results)
            cell_key = f"delta={d}_eta={e}"
            cells[cell_key] = {
                "delta": d,
                "eta": e,
                "trans_err":     arr[:, 0].tolist(),
                "single_err":    arr[:, 1].tolist(),
                "adaptive_err":  arr[:, 2].tolist(),
                "tau_star":      arr[:, 3].tolist(),
                "theta_star":    arr[:, 4].tolist(),
                "trans_mean":    float(arr[:, 0].mean()),
                "single_mean":   float(arr[:, 1].mean()),
                "adaptive_mean": float(arr[:, 2].mean()),
            }
            print(f"  delta={d:.2f} eta={e:.2f}  Trans-RR={cells[cell_key]['trans_mean']:.4f}  "
                  f"Single-RR={cells[cell_key]['single_mean']:.4f}  Adaptive={cells[cell_key]['adaptive_mean']:.4f}")

    Path("res").mkdir(exist_ok=True)
    filename = f"res/2sensitivity_delta_eta_p{p}_simu{K}.json"
    payload = {
        "p": p, "n": n, "K": K, "dd": dd,
        "delta_grid": delta_grid,
        "eta_grid": eta_grid,
        "cells": cells,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"Saved: {filename}")
    print(f"Total wall time: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
