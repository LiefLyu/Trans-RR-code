"""Standalone driver for §4.3 Case II (Cauchy) main comparison.

7 h values x M reps x 4 ridge methods (Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR;
Lasso variants are excluded under heavy-tailed errors per the paper's protocol).
Output: res/2cauchy_cv_results_p400_simu{M}.json with the schema documented
in run_gaussian_main.py (errs, tau_st/src/tgt/pool, theta_star).
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
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from transrr_lib._grids import TAU_GRID
from pipelines import estimate_regression_models_noLASSO


def compute_one_rep(i, p, n, nn, beta_0, w_0, sigma, sigma1,
                    psi_delta, psi_eta, tau_range):
    np.random.seed(i)
    # Scale-mixture lambda_i ~ Unif(0, sqrt(3)) so E[lambda^2] = 1
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
    keys = ["single_robust_ridge", "transfer_robust_ridge", "adaptive", "pooled_robust_ridge"]
    errnorms = tuple(np.sum((res[k]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2) for k in keys)
    taus = (
        float(res["single_robust_ridge"]["optimal_tau"]),
        float(res["transfer_robust_ridge"]["optimal_tau_source"]),
        float(res["transfer_robust_ridge"]["optimal_tau_target_diff"]),
        float(res["pooled_robust_ridge"]["optimal_tau"]),
    )
    theta_star = float(res["adaptive"]["theta_star"])
    return errnorms + taus + (theta_star,)  # 4 errs + 4 taus + 1 theta = 9


def run_dd(p, n, K, dd, n_jobs, tau_range):
    nn = n * 2
    sigma, sigma1 = 1, 2
    psi_delta, psi_eta = 1.35, 0.1

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_rep)(
            i, p, n, nn, beta_0, w_0, sigma, sigma1, psi_delta, psi_eta, tau_range
        )
        for i in tqdm(range(K), desc=f"dd={dd:.3f}")
    )
    arr = np.array(results_list)             # K x 9
    errnorms = arr[:, :4]
    tau_values = arr[:, 4:8]
    theta_values = arr[:, 8]
    mean_err = np.nanmean(errnorms, axis=0)
    std_err = np.nanstd(errnorms, axis=0)
    cols = ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"]
    return mean_err, std_err, pd.DataFrame(errnorms, columns=cols), tau_values, theta_values


def main():
    p_val = 400
    n_val = 400
    K_val = 500
    n_jobs_val = 11

    tau_range_val = TAU_GRID
    dd_values = np.power(np.e, np.arange(-2.0, 1.5, 0.5))

    print(f"=== Cauchy main comparison ===")
    print(f"p={p_val}, n={n_val}, K={K_val}, n_jobs={n_jobs_val}")
    print(f"tau_range: {[f'{x:.3g}' for x in tau_range_val]}")
    print(f"dd_values: {[f'{x:.4f}' for x in dd_values]}")
    print()

    all_results = {}
    t0 = time.time()
    for dd_val in dd_values:
        mean_err, std_err, df, tau_values, theta_values = run_dd(
            p=p_val, n=n_val, K=K_val, dd=dd_val,
            n_jobs=n_jobs_val, tau_range=tau_range_val,
        )
        all_results[dd_val] = {
            "mean_err":     mean_err,
            "std_err":      std_err,
            "results_df":   df,
            "tau_values":   tau_values,
            "theta_values": theta_values,
        }
        print(f"  dd={dd_val:.4f}  mean = {[f'{x:.4f}' for x in mean_err]}")

    Path("res").mkdir(exist_ok=True)
    filename = f"res/2cauchy_cv_results_p{p_val}_simu{K_val}.json"
    converted = {
        str(dd_val): {
            "mean_err":   item["mean_err"].tolist(),
            "std_err":    item["std_err"].tolist(),
            "results_df": item["results_df"].to_dict(),
            "tau_st":     item["tau_values"][:, 0].tolist(),
            "tau_src":    item["tau_values"][:, 1].tolist(),
            "tau_tgt":    item["tau_values"][:, 2].tolist(),
            "tau_pool":   item["tau_values"][:, 3].tolist(),
            "theta_star": item["theta_values"].tolist(),
        }
        for dd_val, item in all_results.items()
    }
    with open(filename, "w") as f:
        json.dump(converted, f, indent=4)

    print()
    print(f"Saved: {filename}")
    print(f"Total wall time: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
