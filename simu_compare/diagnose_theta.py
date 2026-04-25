"""Theta diagnostic experiment: when does Adaptive pick interior theta?

Theory (MSE criterion):
  L(theta) = (1/n) * sum_i (theta * r_t(i) + (1 - theta) * r_s(i))^2
where r_t = y - pred_trans, r_s = y - pred_single (both fold-level).
This is a 1-D quadratic in theta, minimised on [0, 1] at
  theta*_uncon = <r_s, r_s - r_t> / ||r_s - r_t||^2
which lies in (0, 1) when ||r_t||/||r_s|| ~ 1 and r_t, r_s are not parallel.

The §4.3 sweep over h is exactly the right diagnostic: at small h Trans-RR
strictly dominates (theta -> 1), at large h Single-RR strictly dominates
(theta -> 0), and the transition region around h ~ 1 is where the two
methods are comparable and should produce interior theta.

This script runs Case I (Gaussian) at K=100 reps for each of 6 h values and
collects theta_star per rep. Output: res/2theta_diagnostic.json.
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


def worker(seed, p, n, nn, beta_0, w_0, sigma, sigma1, tau_range):
    np.random.seed(seed)
    X = np.random.normal(size=(n, p))
    Y = X @ beta_0 + np.random.normal(0, sigma, n)
    X1 = np.random.normal(size=(nn, p))
    Y1 = X1 @ w_0 + np.random.normal(0, sigma1, nn)
    res = estimate_regression_models_noLASSO(
        X, Y, X1, Y1,
        delta_param=1.35, eta_param=0.1,
        tau_range=tau_range,
    )
    err_sr  = float(np.sum((res["single_robust_ridge"]["betahat"]   - beta_0) ** 2) / np.sum(beta_0 ** 2))
    err_tr  = float(np.sum((res["transfer_robust_ridge"]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2))
    err_ada = float(np.sum((res["adaptive"]["betahat"]              - beta_0) ** 2) / np.sum(beta_0 ** 2))
    err_pr  = float(np.sum((res["pooled_robust_ridge"]["betahat"]   - beta_0) ** 2) / np.sum(beta_0 ** 2))
    theta = float(res["adaptive"]["theta_star"])
    return err_sr, err_tr, err_ada, err_pr, theta


def main():
    p = 400
    n = 400
    nn = 800
    K = 100
    n_jobs = 4  # gaussian sim is using 8 cores; leave 4 for this diagnostic to avoid oversubscription

    sigma, sigma1 = 1, 2
    tau_range = np.logspace(0, 1, 10, base=3)

    # h values picked to span small / transition / large
    dd_values = [
        float(np.exp(-2.0)),  # h = 0.135  (small, expect theta~1)
        float(np.exp(-1.0)),  # h = 0.368
        float(np.exp(-0.5)),  # h = 0.607
        1.0,                  # h = 1.000  (transition; expect interior most likely)
        float(np.exp(0.5)),   # h = 1.649
        float(np.exp(1.0)),   # h = 2.718  (large, expect theta~0)
    ]

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)

    print("=== Theta diagnostic experiment ===")
    print(f"p={p}, n={n}, nn={nn}, K={K}, n_jobs={n_jobs}")
    print(f"dd values: {[f'{x:.4f}' for x in dd_values]}")
    print()

    payload = {"p": p, "n": n, "nn": nn, "K": K, "cells": []}
    t0 = time.time()
    for dd in dd_values:
        delta_0 = np.ones(p) * dd / np.sqrt(p)
        w_0 = beta_0 - delta_0
        results = Parallel(n_jobs=n_jobs)(
            delayed(worker)(i, p, n, nn, beta_0, w_0, sigma, sigma1, tau_range)
            for i in tqdm(range(K), desc=f"dd={dd:.3f}")
        )
        arr = np.array(results)
        err_sr, err_tr, err_ada, err_pr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        thetas = arr[:, 4]
        n_interior = int(np.sum((thetas > 0.0) & (thetas < 1.0)))
        n_at_one = int(np.sum(thetas == 1.0))
        n_at_zero = int(np.sum(thetas == 0.0))
        cell = {
            "dd": float(dd),
            "single_err":   err_sr.tolist(),
            "trans_err":    err_tr.tolist(),
            "adaptive_err": err_ada.tolist(),
            "pooled_err":   err_pr.tolist(),
            "theta_star":   thetas.tolist(),
            "summary": {
                "single_mean":   float(err_sr.mean()),
                "trans_mean":    float(err_tr.mean()),
                "adaptive_mean": float(err_ada.mean()),
                "pooled_mean":   float(err_pr.mean()),
                "theta_mean":    float(thetas.mean()),
                "theta_median":  float(np.median(thetas)),
                "n_interior":    n_interior,
                "n_at_one":      n_at_one,
                "n_at_zero":     n_at_zero,
                "K":             K,
            },
        }
        payload["cells"].append(cell)
        s = cell["summary"]
        print(
            f"  dd={dd:.3f}  "
            f"SR={s['single_mean']:.4f} TR={s['trans_mean']:.4f} Ada={s['adaptive_mean']:.4f}  "
            f"theta: mean={s['theta_mean']:.2f} median={s['theta_median']:.1f}  "
            f"interior={n_interior}/{K}, at-1={n_at_one}, at-0={n_at_zero}"
        )

    Path("res").mkdir(exist_ok=True)
    fn = "res/2theta_diagnostic.json"
    with open(fn, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved: {fn}")
    print(f"Total wall time: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
