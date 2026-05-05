"""Standalone driver for §4.3 Case I (Gaussian) main comparison.

7 h values x M reps x 6 methods (incl. Trans-RR-Ada).
Output: res/2gaussian_cv_results_p400_simu{M}.json with the following schema
per dd-key:
    {
        "mean_err":   [6 floats],   # mean errnorm per method
        "std_err":    [6 floats],   # std errnorm per method
        "results_df": {...},        # M x 6 errnorm DataFrame as dict
        "tau_st":     [M floats],   # Single-RR's selected tau per rep
        "tau_src":    [M floats],   # Trans-RR source-step tau per rep
        "tau_tgt":    [M floats],   # Trans-RR target-step tau per rep
        "tau_pool":   [M floats],   # Pooled-RR tau per rep
        "theta_star": [M floats],   # Trans-RR-Ada theta per rep
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
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Make transrr_lib importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from transrr_lib._grids import TAU_GRID
from pipelines import estimate_regression_models


def compute_one_rep(i, p, n, nn, beta_0, w_0, sigma, sigma1,
                    psi_delta, psi_eta, tau_range, alphas_c):
    np.random.seed(i)
    X = np.random.normal(size=(n, p))
    Y = X @ beta_0 + np.random.normal(0, sigma, n)
    X1 = np.random.normal(size=(nn, p))
    Y1 = X1 @ w_0 + np.random.normal(0, sigma1, nn)

    res = estimate_regression_models(
        X, Y, X1, Y1,
        delta_param=psi_delta, eta_param=psi_eta,
        tau_range=tau_range, alphas_custom=alphas_c,
    )

    keys = ["single_robust_ridge", "transfer_robust_ridge", "adaptive",
            "pooled_robust_ridge", "single_task_lasso", "transfer_lasso"]
    errnorms = tuple(
        np.sum((res[k]["betahat"] - beta_0) ** 2) / np.sum(beta_0 ** 2)
        for k in keys
    )
    taus = (
        float(res["single_robust_ridge"]["optimal_tau"]),
        float(res["transfer_robust_ridge"]["optimal_tau_source"]),
        float(res["transfer_robust_ridge"]["optimal_tau_target_diff"]),
        float(res["pooled_robust_ridge"]["optimal_tau"]),
    )
    theta_star = float(res["adaptive"]["theta_star"])
    return errnorms + taus + (theta_star,)  # 6 errnorms + 4 taus + 1 theta = 11


def run_dd(p, n, K, dd, n_jobs, tau_range, alphas_c):
    nn = n * 2
    sigma, sigma1 = 1, 2
    psi_delta, psi_eta = 1.35, 0.1

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p)
    beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_rep)(
            i, p, n, nn, beta_0, w_0, sigma, sigma1,
            psi_delta, psi_eta, tau_range, alphas_c
        )
        for i in tqdm(range(K), desc=f"dd={dd:.3f}")
    )
    arr = np.array(results_list)            # K x 11
    errnorms = arr[:, :6]                    # K x 6
    tau_values = arr[:, 6:10]                # K x 4 (st, src, tgt, pool)
    theta_values = arr[:, 10]                # K
    mean_err = np.nanmean(errnorms, axis=0)
    std_err = np.nanstd(errnorms, axis=0)
    cols = ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR", "Single Lasso", "Trans Lasso"]
    return mean_err, std_err, pd.DataFrame(errnorms, columns=cols), tau_values, theta_values


def main():
    p_val = 400
    n_val = 400
    K_val = 500
    n_jobs_val = 11

    tau_range_val = TAU_GRID
    alpha_range_val = None
    dd_values = np.power(np.e, np.arange(-2.0, 1.5, 0.5))  # 7 values

    print(f"=== Gaussian main comparison ===")
    print(f"p={p_val}, n={n_val}, K={K_val}, n_jobs={n_jobs_val}")
    print(f"BLAS env: VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS')} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"tau_range: {[f'{x:.3g}' for x in tau_range_val]}")
    print(f"dd_values: {[f'{x:.4f}' for x in dd_values]}")
    print()

    all_results = {}
    t0 = time.time()
    for dd_val in dd_values:
        mean_err, std_err, errnorm_df, tau_values, theta_values = run_dd(
            p=p_val, n=n_val, K=K_val, dd=dd_val,
            n_jobs=n_jobs_val, tau_range=tau_range_val, alphas_c=alpha_range_val,
        )
        all_results[dd_val] = {
            "mean_err":     mean_err,
            "std_err":      std_err,
            "results_df":   errnorm_df,
            "tau_values":   tau_values,
            "theta_values": theta_values,
        }
        print(f"  dd={dd_val:.4f}  mean = {[f'{x:.4f}' for x in mean_err]}")

    Path("res").mkdir(exist_ok=True)
    filename = f"res/2gaussian_cv_results_p{p_val}_simu{K_val}.json"
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
