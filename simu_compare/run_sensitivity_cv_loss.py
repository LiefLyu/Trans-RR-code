"""Reviewer 2 Comment 3 sensitivity reruns:
(i)  CV criterion: MAE (default) -> MSE       [Tab B2]
(ii) Robust loss: smoothed_huber (default) -> pseudo_huber  [Tab B3]

Reruns Cases I/II/III at all 7 h values (matching the §4.3 main figure)
with M reps per cell. Output a single JSON containing both sweeps.

Methods: Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR.

Output: res/2sensitivity_cv_loss_p400_simu{M}.json
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

from transrr_lib._grids import TAU_GRID
from pipelines import estimate_regression_models_noLASSO


def make_data_gaussian(seed, p, n, nn, beta_0, w_0, sigma, sigma1):
    np.random.seed(seed)
    X = np.random.normal(size=(n, p))
    Y = X @ beta_0 + np.random.normal(0, sigma, n)
    X1 = np.random.normal(size=(nn, p))
    Y1 = X1 @ w_0 + np.random.normal(0, sigma1, nn)
    return X, Y, X1, Y1


def make_data_cauchy(seed, p, n, nn, beta_0, w_0, sigma, sigma1):
    np.random.seed(seed)
    lam = np.random.uniform(0, np.sqrt(3), n)
    X = np.random.normal(size=(n, p)) * lam[:, np.newaxis]
    Y = X @ beta_0 + np.random.standard_cauchy(size=n) * sigma
    lam1 = np.random.uniform(0, np.sqrt(3), nn)
    X1 = np.random.normal(size=(nn, p)) * lam1[:, np.newaxis]
    Y1 = X1 @ w_0 + np.random.standard_cauchy(size=nn) * sigma1
    return X, Y, X1, Y1


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
    X1 = np.vstack((X21, X22)); Y1 = np.concatenate((Y21, Y22))
    return X, Y, X1, Y1


DATA_GENERATORS = {
    "gaussian": make_data_gaussian,
    "cauchy": make_data_cauchy,
    "mix": make_data_mix,
}


def worker(seed, case, p, n, nn, beta_0, w_0, sigma, sigma1,
           tau_range, criterion, loss):
    X, Y, X1, Y1 = DATA_GENERATORS[case](seed, p, n, nn, beta_0, w_0, sigma, sigma1)
    res = estimate_regression_models_noLASSO(
        X, Y, X1, Y1,
        delta_param=1.35, eta_param=0.1,
        tau_range=tau_range,
        criterion=criterion, loss=loss,
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


def run_one_cell(case, dd, criterion, loss, p, n, K, n_jobs, tau_range):
    nn = n * 2
    sigma, sigma1 = 1, 2

    rng = np.random.RandomState(1)
    beta_0 = rng.uniform(size=p); beta_0 /= np.linalg.norm(beta_0, 2)
    delta_0 = np.ones(p) * dd / np.sqrt(p)
    w_0 = beta_0 - delta_0

    desc = f"[{case} crit={criterion} loss={loss}] dd={dd:.3f}"
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            i, case, p, n, nn, beta_0, w_0, sigma, sigma1,
            tau_range, criterion, loss,
        )
        for i in tqdm(range(K), desc=desc)
    )
    arr = np.array(results_list)
    errnorm = arr[:, :4]
    tau_values = arr[:, 4:8]
    thetas = arr[:, 8]
    return {
        "case":         case,
        "dd":           float(dd),
        "criterion":    criterion,
        "loss":         loss,
        "mean_err":     errnorm.mean(axis=0).tolist(),
        "std_err":      errnorm.std(axis=0).tolist(),
        "errs": {
            "Single RR":     errnorm[:, 0].tolist(),
            "Trans RR":      errnorm[:, 1].tolist(),
            "Trans-RR-Ada":  errnorm[:, 2].tolist(),
            "Pooled RR":     errnorm[:, 3].tolist(),
        },
        "tau_st":   tau_values[:, 0].tolist(),
        "tau_src":  tau_values[:, 1].tolist(),
        "tau_tgt":  tau_values[:, 2].tolist(),
        "tau_pool": tau_values[:, 3].tolist(),
        "theta_star": thetas.tolist(),
        "method_names": ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"],
    }


def main():
    p = 400
    n = 400
    K = 500
    n_jobs = 11

    tau_range = TAU_GRID

    cases = ["gaussian", "cauchy", "mix"]
    dd_values = list(np.power(np.e, np.arange(-2.0, 1.5, 0.5)))   # 7 h values

    sweeps = [
        # (criterion, loss, label)
        ("mse", "smoothed_huber", "mse_smoothed_huber"),
        ("mae", "pseudo_huber",   "mae_pseudo_huber"),
    ]

    print("=== CV-criterion / loss sensitivity reruns ===")
    print(f"p={p}, n={n}, K={K}, n_jobs={n_jobs}")
    print(f"tau_range: {[f'{x:.3g}' for x in tau_range]}")
    print(f"cases: {cases}, dd values: {[f'{x:.4f}' for x in dd_values]}")
    print(f"sweeps: {[s[2] for s in sweeps]}")
    print()

    payload = {
        "p": p, "n": n, "K": K,
        "tau_grid": [float(x) for x in tau_range],
        "sweeps": {},
    }
    t0 = time.time()
    for criterion, loss, label in sweeps:
        cells = []
        for case in cases:
            for dd in dd_values:
                cell = run_one_cell(
                    case=case, dd=dd, criterion=criterion, loss=loss,
                    p=p, n=n, K=K, n_jobs=n_jobs, tau_range=tau_range,
                )
                cells.append(cell)
                print(f"  [{label}] case={case} dd={dd:.4f}  mean = {[f'{x:.4f}' for x in cell['mean_err']]}")
        payload["sweeps"][label] = cells

    Path("res").mkdir(exist_ok=True)
    filename = f"res/2sensitivity_cv_loss_p{p}_simu{K}.json"
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"Saved: {filename}")
    print(f"Total wall time: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
