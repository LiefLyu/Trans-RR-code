"""Reviewer 2 Comment 1: AR(1) correlated-predictors simulation.

Tests robustness of qualitative findings (positive vs negative transfer regimes,
Adaptive selector behavior) when cov(x_i) is not identity. Uses Case I (Gaussian
errors) only since the sensitivity to design covariance is the focus.

Setup: cov(x_i) = Σ with Σ_{jk} = ρ^|j−k|, ρ ∈ {0.3, 0.6}.
Methods: Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR (no Lasso, since we
already established under non-sparse coefficients Lasso is dominated).
1000 reps × 7 h × 2 ρ values.

Output: res/2corr_ar1_results_p400_simu1000.json (one entry per (rho, dd) pair)
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
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

from pipelines import estimate_regression_models_noLASSO


def ar1_cholesky(p, rho):
    """Return lower-triangular L such that L L^T = Σ_AR(1) with Σ_{jk} = rho^|j-k|."""
    Sigma = rho ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    return np.linalg.cholesky(Sigma)


def compute_errnorm_ar1(i, p, n, nn, beta_0, w_0, sigma, sigma1, L,
                        psi_delta=1.35, psi_eta=0.1, tau_range=None):
    np.random.seed(i)
    # X_i ~ N(0, Σ) via X = Z @ L^T  (with Z having i.i.d. N(0,1) entries)
    Z = np.random.normal(size=(n, p))
    X = Z @ L.T
    Y = X @ beta_0 + np.random.normal(0, sigma, n)

    Z1 = np.random.normal(size=(nn, p))
    X1 = Z1 @ L.T
    Y1 = X1 @ w_0 + np.random.normal(0, sigma1, nn)

    if tau_range is None:
        tau_range = np.logspace(-4, 1, 15)

    res = estimate_regression_models_noLASSO(
        X, Y, X1, Y1,
        delta_param=psi_delta, eta_param=psi_eta,
        tau_range=tau_range,
    )
    keys = ["single_robust_ridge", "transfer_robust_ridge", "adaptive", "pooled_robust_ridge"]
    errnorms = tuple(np.sum((res[k]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2) for k in keys)
    return errnorms + (float(res["adaptive"]["theta_star"]),)  # 4 errnorms + 1 theta_star


def run_dd(p, n, K, dd, rho, n_jobs, tau_range):
    nn = n * 2
    sigma, sigma1 = 1, 2
    psi_delta, psi_eta = 1.35, 0.1

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0
    L = ar1_cholesky(p, rho)

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_errnorm_ar1)(
            i, p, n, nn, beta_0, w_0, sigma, sigma1, L, psi_delta, psi_eta, tau_range
        )
        for i in tqdm(range(K), desc=f"rho={rho:.1f} dd={dd:.3f}")
    )

    arr = np.array(results_list)
    errnorm = arr[:, :4]
    theta_values = arr[:, 4]
    mean_err = np.nanmean(errnorm, axis=0)
    std_err = np.nanstd(errnorm, axis=0)
    cols = ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"]
    return mean_err, std_err, pd.DataFrame(errnorm, columns=cols), theta_values


def main():
    p_val = 400
    n_val = 400
    K_val = 1000
    n_jobs_val = 8

    tau_range_val = np.logspace(0, 1, 10, base=3)
    rho_values = [0.3, 0.6]
    dd_values = np.power(np.e, np.arange(-2.0, 1.5, 0.5))

    print(f"=== AR(1) correlated-predictors comparison ===")
    print(f"p={p_val}, n={n_val}, K={K_val}, n_jobs={n_jobs_val}")
    print(f"rho values: {rho_values}")
    print(f"dd values:  {[f'{x:.4f}' for x in dd_values]}")
    print()

    all_results = {}
    t0 = time.time()
    for rho in rho_values:
        for dd_val in dd_values:
            mean_err, std_err, df, thetas = run_dd(
                p=p_val, n=n_val, K=K_val, dd=dd_val, rho=rho,
                n_jobs=n_jobs_val, tau_range=tau_range_val,
            )
            key = f"rho={rho}_dd={dd_val}"
            all_results[key] = {
                "rho": rho,
                "dd": float(dd_val),
                "mean_err": mean_err,
                "std_err": std_err,
                "results_df": df,
                "theta_star": thetas.tolist(),
            }
            print(f"  rho={rho} dd={dd_val:.4f}  mean_err = {[f'{x:.4f}' for x in mean_err]}")

    Path("res").mkdir(exist_ok=True)
    filename = f"res/2corr_ar1_results_p{p_val}_simu{K_val}.json"
    converted = {
        key: {
            "rho":         item["rho"],
            "dd":          item["dd"],
            "mean_err":    item["mean_err"].tolist(),
            "std_err":     item["std_err"].tolist(),
            "results_df":  item["results_df"].to_dict(),
            "theta_star":  item["theta_star"],
        }
        for key, item in all_results.items()
    }
    with open(filename, "w") as f:
        json.dump(converted, f, indent=4)

    print()
    print(f"Saved: {filename}")
    print(f"Total wall time: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
