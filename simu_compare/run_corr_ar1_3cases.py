"""§4.4 (revised): AR(1) covariance robustness across all three error cases.

Replaces the previous Case I x {rho=0.3, 0.6} design with {Case I, II, III}
x rho=0.6, mirroring the structure of Figure 3 (figmethods).

Reviewer 2 Comment 1 motivated this: the theory assumes cov(x_i) = I_p,
so we want to verify that the negative-transfer transition observed in
Section 4.3 extends beyond the identity-covariance assumption. The
revised design tests the qualitative pattern under AR(1) covariance for
all three noise/design cases (Gaussian, Cauchy, Mixture), giving a more
direct sensitivity check than testing two correlation levels under one
case.

Setup: 3 cases x 7 h x rho=0.6 x M reps x 4 ridge methods (no Lasso).
Data generation under AR(1) covariance Sigma_{jk} = rho^|j-k|:
    Case I:   X = Z @ L^T        (Z ~ N(0, I_p))
    Case II:  X = lam * (Z @ L^T)  (lam ~ Unif(0, sqrt(3)) per row;
                                    scale mixture with E[lam^2] = 1)
    Case III: half = Case II AR(1) + half = Case I AR(1)
Errors:
    Case I:   N(0, sigma^2)
    Case II:  Cauchy(0, sigma)
    Case III: half = Cauchy + half = Gaussian (matched to half above)

Output: res/2corr_ar1_3cases_p400_simu{M}.json with the per-cell schema
documented in run_gaussian_main.py (mean_err, std_err, tau_st/src/tgt/pool,
theta_star). The previous res/2corr_ar1_results_p400_simu500.json is
preserved as a backup of the old 2-rho design.
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


def ar1_cholesky(p, rho):
    Sigma = rho ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    return np.linalg.cholesky(Sigma)


# ---- Data generators under AR(1) covariance ----
def make_data_gaussian_ar1(seed, p, n, nn, beta_0, w_0, sigma, sigma1, L):
    """Case I + AR(1)."""
    np.random.seed(seed)
    Z = np.random.normal(size=(n, p))
    X = Z @ L.T
    Y = X @ beta_0 + np.random.normal(0, sigma, n)

    Z1 = np.random.normal(size=(nn, p))
    X1 = Z1 @ L.T
    Y1 = X1 @ w_0 + np.random.normal(0, sigma1, nn)
    return X, Y, X1, Y1


def make_data_cauchy_ar1(seed, p, n, nn, beta_0, w_0, sigma, sigma1, L):
    """Case II + AR(1): scale-mixture row scaling on top of AR(1) within each row."""
    np.random.seed(seed)
    lam = np.random.uniform(0, np.sqrt(3), n)
    Z = np.random.normal(size=(n, p))
    X = lam[:, None] * (Z @ L.T)
    Y = X @ beta_0 + np.random.standard_cauchy(size=n) * sigma

    lam1 = np.random.uniform(0, np.sqrt(3), nn)
    Z1 = np.random.normal(size=(nn, p))
    X1 = lam1[:, None] * (Z1 @ L.T)
    Y1 = X1 @ w_0 + np.random.standard_cauchy(size=nn) * sigma1
    return X, Y, X1, Y1


def make_data_mix_ar1(seed, p, n, nn, beta_0, w_0, sigma, sigma1, L):
    """Case III + AR(1): half scale-mixture+Cauchy (Case II AR(1)), half pure
    Gaussian (Case I AR(1))."""
    np.random.seed(seed)
    # Target: half Case II AR(1)
    lam = np.random.uniform(0, np.sqrt(3), n // 2)
    Z11 = np.random.normal(size=(n // 2, p))
    X11 = lam[:, None] * (Z11 @ L.T)
    Y11 = X11 @ beta_0 + np.random.standard_cauchy(size=n // 2)
    # Target: half Case I AR(1)
    Z12 = np.random.normal(size=(n // 2, p))
    X12 = Z12 @ L.T
    Y12 = X12 @ beta_0 + np.random.normal(0, sigma, n // 2)

    X = np.vstack((X11, X12))
    Y = np.concatenate((Y11, Y12))

    # Source: same mix structure
    lam1 = np.random.uniform(0, np.sqrt(3), nn // 2)
    Z21 = np.random.normal(size=(nn // 2, p))
    X21 = lam1[:, None] * (Z21 @ L.T)
    Y21 = X21 @ w_0 + np.random.standard_cauchy(size=nn // 2) * sigma1

    Z22 = np.random.normal(size=(nn // 2, p))
    X22 = Z22 @ L.T
    Y22 = X22 @ w_0 + np.random.normal(0, sigma1, nn // 2)

    X1 = np.vstack((X21, X22))
    Y1 = np.concatenate((Y21, Y22))
    return X, Y, X1, Y1


DATA_GENERATORS = {
    "gaussian": make_data_gaussian_ar1,
    "cauchy":   make_data_cauchy_ar1,
    "mix":      make_data_mix_ar1,
}


def compute_one_rep(seed, case, p, n, nn, beta_0, w_0, sigma, sigma1, L,
                    psi_delta, psi_eta, tau_range):
    X, Y, X1, Y1 = DATA_GENERATORS[case](
        seed, p, n, nn, beta_0, w_0, sigma, sigma1, L
    )
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


def run_dd(case, p, n, K, dd, rho, n_jobs, tau_range):
    nn = n * 2
    sigma, sigma1 = 1, 2
    psi_delta, psi_eta = 1.35, 0.1

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0
    L = ar1_cholesky(p, rho)

    desc = f"{case} rho={rho:.1f} dd={dd:.3f}"
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_one_rep)(
            i, case, p, n, nn, beta_0, w_0, sigma, sigma1, L,
            psi_delta, psi_eta, tau_range,
        )
        for i in tqdm(range(K), desc=desc)
    )
    arr = np.array(results_list)
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
    rho_val = 0.6
    dd_values = np.power(np.e, np.arange(-2.0, 1.5, 0.5))
    cases = ["gaussian", "cauchy", "mix"]

    print(f"=== AR(1) covariance robustness across 3 cases ===")
    print(f"p={p_val}, n={n_val}, K={K_val}, n_jobs={n_jobs_val}, rho={rho_val}")
    print(f"tau_range: {[f'{x:.3g}' for x in tau_range_val]}")
    print(f"cases: {cases}")
    print(f"dd values: {[f'{x:.4f}' for x in dd_values]}")
    print()

    all_results = {}
    t0 = time.time()
    for case in cases:
        for dd_val in dd_values:
            mean_err, std_err, df, tau_values, theta_values = run_dd(
                case=case, p=p_val, n=n_val, K=K_val, dd=dd_val, rho=rho_val,
                n_jobs=n_jobs_val, tau_range=tau_range_val,
            )
            key = f"case={case}_dd={dd_val}"
            all_results[key] = {
                "case":         case,
                "rho":          rho_val,
                "dd":           float(dd_val),
                "mean_err":     mean_err,
                "std_err":      std_err,
                "results_df":   df,
                "tau_values":   tau_values,
                "theta_values": theta_values,
            }
            print(f"  {case} dd={dd_val:.4f}  mean = {[f'{x:.4f}' for x in mean_err]}")

    Path("res").mkdir(exist_ok=True)
    filename = f"res/2corr_ar1_3cases_p{p_val}_simu{K_val}.json"
    converted = {
        key: {
            "case":        item["case"],
            "rho":         item["rho"],
            "dd":          item["dd"],
            "mean_err":    item["mean_err"].tolist(),
            "std_err":     item["std_err"].tolist(),
            "results_df":  item["results_df"].to_dict(),
            "tau_st":      item["tau_values"][:, 0].tolist(),
            "tau_src":     item["tau_values"][:, 1].tolist(),
            "tau_tgt":     item["tau_values"][:, 2].tolist(),
            "tau_pool":    item["tau_values"][:, 3].tolist(),
            "theta_star":  item["theta_values"].tolist(),
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
