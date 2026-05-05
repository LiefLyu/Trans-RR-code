"""Reviewer 2 Comment 3 -- ridge penalty fixed-tau ablation (Tab B5).

Complementary to the wide-grid sensitivity in run_sensitivity_tau.py (Tab B4):
B4 tests "do conclusions depend on the CV search range?", while this experiment
tests "do conclusions depend on the actual tau value, regardless of CV?"

Setup: 3 cases x 7 h x 4 fixed tau values x 4 ridge methods x M reps.

Fixed tau values (all members of TAU_GRID):
    tau ∈ {1/3, 1, 3, 9}
spanning roughly the IQR of CV-selected tau seen in the diagnostic.

For each cell, all four ridge penalties (Single-RR's tau_st, Trans-RR's tau_1
and tau, Pooled-RR's tau_p) are forced to the same fixed value tau_fixed.
The Adaptive layer still selects theta via CV on the target sample (Algorithm
2 in the paper), so it is meaningful here.

Output: res/2sensitivity_fixed_tau_p400_simu{M}.json with the per-cell schema:
    {
        "case", "dd", "tau_fixed",
        "mean_err":   [Single, Trans, Ada, Pooled],
        "std_err":    [Single, Trans, Ada, Pooled],
        "errs":       per-method per-rep arrays,
        "theta_star": per-rep arrays.
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
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from transrr_lib.robust_ridge_optimizer import solve_robust_ridge
from transrr_lib.adaptive import aggregate_by_cv

_ADAPTIVE_KFOLD_SEED = 2  # matches pipelines.py


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


def fit_ridge_huber(X, y, tau, psi_delta, psi_eta):
    """Closed-form ridge initial guess + Huber-ridge L-BFGS-B refit."""
    n, p = X.shape
    init = np.linalg.solve(
        X.T @ X / n + tau * np.eye(p),
        X.T @ y / n,
    )
    return solve_robust_ridge(
        X, y, tau, psi_delta, psi_eta,
        initial_beta=init, loss='smoothed_huber',
    )


def compute_one_rep(seed, case, p, n, nn, beta_0, w_0, sigma, sigma1,
                    tau_fixed, psi_delta, psi_eta):
    """Fit Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR with all ridge
    penalties forced to tau_fixed (no tau-CV). Adaptive's theta is still
    CV-selected on an independent KFold(seed=2) partition."""
    X, y, X1, y1 = DATA_GENERATORS[case](seed, p, n, nn, beta_0, w_0, sigma, sigma1)

    # 1. Single-RR on (X, y)
    beta_sr = fit_ridge_huber(X, y, tau_fixed, psi_delta, psi_eta)

    # 2. Trans-RR source step on (X1, y1)
    w_hat = fit_ridge_huber(X1, y1, tau_fixed, psi_delta, psi_eta)

    # 3. Trans-RR target step on (X, y - X w_hat)
    y_adj = y - X @ w_hat
    delta_hat = fit_ridge_huber(X, y_adj, tau_fixed, psi_delta, psi_eta)
    beta_tr = delta_hat + w_hat

    # 4. Pooled-RR on stacked design
    XX = np.vstack((X, X1))
    YY = np.concatenate((y, y1))
    beta_pr = fit_ridge_huber(XX, YY, tau_fixed, psi_delta, psi_eta)

    # 5. Adaptive layer (theta-CV on independent KFold(seed=2))
    kf_ada = KFold(n_splits=5, shuffle=True, random_state=_ADAPTIVE_KFOLD_SEED)
    fold_indices = list(kf_ada.split(X))
    fold_betas_single = []
    fold_betas_trans = []
    for train_idx, _ in fold_indices:
        X_tr, y_tr = X[train_idx], y[train_idx]
        # Fold-level Single-RR at tau_fixed
        fold_betas_single.append(
            fit_ridge_huber(X_tr, y_tr, tau_fixed, psi_delta, psi_eta)
        )
        # Fold-level Trans-RR target step at tau_fixed (with full-data w_hat held fixed)
        y_adj_tr = y_tr - X_tr @ w_hat
        delta_fold = fit_ridge_huber(X_tr, y_adj_tr, tau_fixed, psi_delta, psi_eta)
        fold_betas_trans.append(delta_fold + w_hat)

    ada_res = aggregate_by_cv(
        fold_indices=fold_indices,
        fold_betas_trans=fold_betas_trans,
        fold_betas_single=fold_betas_single,
        X=X, Y=y,
        beta_trans_full=beta_tr,
        beta_single_full=beta_sr,
        criterion='mae',
    )
    beta_ada = ada_res['beta_ada']
    theta_star = float(ada_res['theta_star'])

    norm_sq = float(np.sum(beta_0 ** 2))
    err_sr = float(np.sum((beta_sr - beta_0) ** 2) / norm_sq)
    err_tr = float(np.sum((beta_tr - beta_0) ** 2) / norm_sq)
    err_ada = float(np.sum((beta_ada - beta_0) ** 2) / norm_sq)
    err_pr = float(np.sum((beta_pr - beta_0) ** 2) / norm_sq)

    return (err_sr, err_tr, err_ada, err_pr, theta_star)


def run_one_cell(case, dd, tau_fixed, p, n, K, n_jobs):
    nn = 2 * n
    sigma, sigma1 = 1, 2
    psi_delta, psi_eta = 1.35, 0.1

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0

    desc = f"{case} h={dd:.3f} tau={tau_fixed:.3g}"
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_rep)(
            seed, case, p, n, nn, beta_0, w_0, sigma, sigma1,
            tau_fixed, psi_delta, psi_eta,
        )
        for seed in tqdm(range(K), desc=desc)
    )
    arr = np.array(results)
    errnorms = arr[:, :4]
    thetas = arr[:, 4]
    return {
        "case":      case,
        "dd":        float(dd),
        "tau_fixed": float(tau_fixed),
        "mean_err":  errnorms.mean(axis=0).tolist(),
        "std_err":   errnorms.std(axis=0).tolist(),
        "errs": {
            "Single RR":    errnorms[:, 0].tolist(),
            "Trans RR":     errnorms[:, 1].tolist(),
            "Trans-RR-Ada": errnorms[:, 2].tolist(),
            "Pooled RR":    errnorms[:, 3].tolist(),
        },
        "theta_star": thetas.tolist(),
    }


def main():
    p = 400
    n = 400
    K = 500
    n_jobs = 11

    cases = ["gaussian", "cauchy", "mix"]
    dd_values = list(np.power(np.e, np.arange(-2.0, 1.5, 0.5)))     # 7 h values
    # Fixed tau values: span roughly the CV IQR observed in the oracle/CV diagnostic.
    # All four are members of TAU_GRID.
    tau_fixed_values = [1.0/3.0, 1.0, 3.0, 9.0]

    print("=== Fixed-tau ablation (Tab B5) ===")
    print(f"p={p}, n={n}, K={K}, n_jobs={n_jobs}")
    print(f"cases:    {cases}")
    print(f"h values: {[f'{x:.3f}' for x in dd_values]}")
    print(f"tau_fixed values: {tau_fixed_values}")
    print(f"-> {len(cases) * len(dd_values) * len(tau_fixed_values)} cells x M={K}")
    print()

    cells = []
    t0 = time.time()
    for case in cases:
        for dd in dd_values:
            for tau in tau_fixed_values:
                cell = run_one_cell(case, dd, tau, p, n, K, n_jobs)
                cells.append(cell)
                print(f"  {case} h={dd:.3f} tau={tau:.3g}  mean = {[f'{x:.4f}' for x in cell['mean_err']]}")

    payload = {
        "p": p, "n": n, "K": K,
        "cases": cases,
        "dd_values": [float(x) for x in dd_values],
        "tau_fixed_values": tau_fixed_values,
        "cells": cells,
        "method_names": ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"],
        "wall_time_minutes": (time.time() - t0) / 60.0,
    }
    Path("res").mkdir(exist_ok=True)
    filename = f"res/2sensitivity_fixed_tau_p{p}_simu{K}.json"
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved: {filename}")
    print(f"Total wall time: {payload['wall_time_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
