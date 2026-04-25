#!/usr/bin/env python3
# Auto-generated MAE-CV variant of realdata_run.py (Phase 3a CV-criterion decision).
# Diffs vs MSE version: criterion="mae" inside fit_robust_center_no_intercept,
# output JSON/CSV files have a "_mae" suffix.

import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning

import os, sys
_EXP_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
from transrr_lib.robust_ridge_optimizer import solve_robust_ridge
from transrr_lib.find_tau_opt import find_optimal_tau_robust_ridge

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fixed config (per-split protocol)
RANDOM_STATE = 10
NUM_SAMPLE_ROWS = 160
WHITEN_EPS = 1e-6

N_REPEATS = 20
REPEAT_SEED_START = 1000

# Robust loss shape
DELTA_PARAM = 1.35
ETA_PARAM = 0.1

# Common ridge / lasso grid
COMMON_GRID = np.logspace(-4, 1, 11)

# Adaptive Trans-RR weight grid (Phase 3a)
THETA_GRID = np.linspace(0, 1, 11)

# Final modeling choices
Y_MODE = "center"
USE_INTERCEPT = False

# === Multi-config robustness checks (Reviewer 2 Comment 5) ===
# Each config is a triplet (STEP, OFFSET, WHITEN). All keep STEP=4 to stay in moderate-dim regime.
CONFIGS = [
    {"name": "main",      "STEP": 4, "OFFSET": 0, "WHITEN": True},
    {"name": "off1",      "STEP": 4, "OFFSET": 1, "WHITEN": True},
    {"name": "off2",      "STEP": 4, "OFFSET": 2, "WHITEN": True},
    {"name": "off3",      "STEP": 4, "OFFSET": 3, "WHITEN": True},
    {"name": "no_whiten", "STEP": 4, "OFFSET": 0, "WHITEN": False},
]

print("COMMON_GRID:", COMMON_GRID)
print(f"N_REPEATS={N_REPEATS}, REPEAT_SEED_START={REPEAT_SEED_START}")
print(f"Configs to run: {[c['name'] for c in CONFIGS]}")


# Load raw data
X_df = pd.read_csv("shootout/X.csv")
Xt_df = pd.read_csv("shootout/test_X.csv")
X1_df = pd.read_csv("shootout/X_1.csv")
X1t_df = pd.read_csv("shootout/test_X1.csv")

y_df = pd.read_csv("shootout/y.csv")
yt_df = pd.read_csv("shootout/test_y.csv")

y_train_full = y_df.iloc[:, 2].to_numpy()
y_test_full = yt_df.iloc[:, 2].to_numpy()

X_target_train_raw = X_df.to_numpy()
X_target_test_raw = Xt_df.to_numpy()
X_source_train_raw = X1_df.to_numpy()
X_source_test_raw = X1t_df.to_numpy()

print("Raw shapes:")
print(f"  target train/test: {X_target_train_raw.shape} / {X_target_test_raw.shape}")
print(f"  source train/test: {X_source_train_raw.shape} / {X_source_test_raw.shape}")
print(f"  y train/test:      {y_train_full.shape} / {y_test_full.shape}")


# Preprocessing as a function: select predictors then optionally whiten

def fit_whitener(X, eps=1e-6):
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, eps, None)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return mean, W


def preprocess(STEP, OFFSET, WHITEN, eps=WHITEN_EPS):
    """Select every-STEPth wavelength starting at OFFSET, optionally whiten per domain.
    Returns target/source train/test arrays plus the resulting predictor count.
    """
    p_total = X_target_train_raw.shape[1]
    selected_idx = np.arange(OFFSET, p_total, STEP)

    Xtt_sel = X_target_train_raw[:, selected_idx]
    Xtv_sel = X_target_test_raw[:, selected_idx]
    Xst_sel = X_source_train_raw[:, selected_idx]
    Xsv_sel = X_source_test_raw[:, selected_idx]

    if WHITEN:
        t_mean, t_W = fit_whitener(Xtt_sel, eps)
        s_mean, s_W = fit_whitener(Xst_sel, eps)
        Xtt = (Xtt_sel - t_mean) @ t_W
        Xtv = (Xtv_sel - t_mean) @ t_W
        Xst = (Xst_sel - s_mean) @ s_W
        Xsv = (Xsv_sel - s_mean) @ s_W
    else:
        # Center predictors but do not decorrelate
        t_mean = Xtt_sel.mean(axis=0)
        s_mean = Xst_sel.mean(axis=0)
        Xtt = Xtt_sel - t_mean
        Xtv = Xtv_sel - t_mean
        Xst = Xst_sel - s_mean
        Xsv = Xsv_sel - s_mean

    return Xtt, Xtv, Xst, Xsv, len(selected_idx)


# y center helpers
def fit_y_center(y):
    return {"mu": float(np.mean(y))}


def center_y(y, stats):
    return y - stats["mu"]


def uncenter_y(y_t, stats):
    return y_t + stats["mu"]


# Model helpers (with optional fold-level info for Adaptive) + Adaptive aggregator

def fit_robust_center_no_intercept(X, y, tau_grid, delta, eta, return_folds=False):
    y_stats = fit_y_center(y)
    y_t = center_y(y, y_stats)

    cv_result = find_optimal_tau_robust_ridge(
        X, y_t, tau_range=tau_grid, psi_delta=delta, psi_eta=eta,
        criterion='mae',  # realdata MAE-CV variant for the §5 robustness comparison
        return_folds=return_folds,
    )
    if return_folds:
        tau_opt = cv_result['tau_opt']
    else:
        tau_opt = cv_result

    initial_guess = np.linalg.solve(
        X.T @ X / len(y_t) + tau_opt * np.eye(X.shape[1]),
        X.T @ y_t / len(y_t)
    )
    beta = solve_robust_ridge(X, y_t, tau_opt, delta, eta, initial_beta=initial_guess)

    out = {"beta": beta, "tau": float(tau_opt), "y_stats": y_stats}
    if return_folds:
        out['fold_indices'] = cv_result['fold_indices']
        out['fold_betas']   = cv_result['fold_betas']
    return out


def predict_robust_center_no_intercept(model, X):
    pred_t = X @ model["beta"]
    return uncenter_y(pred_t, model["y_stats"])


def fit_lasso_center_no_intercept(X, y, alpha_grid, random_state):
    y_stats = fit_y_center(y)
    y_t = center_y(y, y_stats)
    lasso = LassoCV(
        fit_intercept=False, max_iter=50000, tol=1e-4,
        alphas=alpha_grid, cv=5, selection="cyclic",
        random_state=random_state,
    ).fit(X, y_t)
    return {"model": lasso, "alpha": float(lasso.alpha_), "y_stats": y_stats}


def predict_lasso_center_no_intercept(model, X):
    pred_t = model["model"].predict(X)
    return uncenter_y(pred_t, model["y_stats"])


def adaptive_select_theta_realdata(
    fold_indices, fold_betas_sr, fold_betas_diff,
    train_X, train_y,
    model_w_beta, mu_train_y1, mu_Y_adjusted, mu_train_y,
    theta_grid=None, criterion='mae',
):
    """Pick theta on the same fold partition by minimising CV residuals of
    theta * pred_trans + (1 - theta) * pred_single, evaluated in original-y space.

    Validation residual at fold k for a candidate theta:
      r_k(theta) = y_train[val_idx]
                   - (1 - theta) * (X[val_idx] @ beta_sr_k + mu_train_y)
                   - theta * (X[val_idx] @ (beta_w + beta_diff_k) + mu_train_y1 + mu_Y_adjusted)
    """
    if theta_grid is None:
        theta_grid = np.linspace(0, 1, 11)

    cv_curve = np.zeros(len(theta_grid))
    for j, theta in enumerate(theta_grid):
        residuals_concat = []
        for k, (_, val_idx) in enumerate(fold_indices):
            X_val = train_X[val_idx]
            pred_sr_k = X_val @ fold_betas_sr[k] + mu_train_y
            pred_tr_k = (
                X_val @ model_w_beta + mu_train_y1
                + X_val @ fold_betas_diff[k] + mu_Y_adjusted
            )
            pred_ada_k = theta * pred_tr_k + (1 - theta) * pred_sr_k
            r = train_y[val_idx] - pred_ada_k
            residuals_concat.append(r ** 2 if criterion == 'mse' else np.abs(r))
        cv_curve[j] = float(np.mean(np.concatenate(residuals_concat)))

    j_star = int(np.argmin(cv_curve))
    return float(theta_grid[j_star]), cv_curve


# One split, one direction (with Adaptive Trans-RR added)

def run_one_split(direction_name, target_train, target_test, source_train, split_seed):
    n_train = min(target_train.shape[0], source_train.shape[0], len(y_train_full))
    n_test = min(target_test.shape[0], len(y_test_full))

    target_train_use = target_train[:n_train]
    source_train_use = source_train[:n_train]
    y_train_use = y_train_full[:n_train]
    target_test_use = target_test[:n_test]
    y_test_use = y_test_full[:n_test]

    rng = np.random.default_rng(split_seed)
    all_indices = np.arange(n_train)
    div_indices = rng.choice(all_indices, size=NUM_SAMPLE_ROWS, replace=False)
    non_div_indices = np.setdiff1d(all_indices, div_indices)

    train_X  = target_train_use[div_indices]
    train_y  = y_train_use[div_indices]
    train_X1 = source_train_use[non_div_indices]
    train_y1 = y_train_use[non_div_indices]
    test_X   = target_test_use
    test_y   = y_test_use

    # 1) Single Robust Ridge (with fold info for Adaptive)
    model_sr = fit_robust_center_no_intercept(
        train_X, train_y, COMMON_GRID, DELTA_PARAM, ETA_PARAM, return_folds=True,
    )
    pred_sr = predict_robust_center_no_intercept(model_sr, test_X)

    # 2) Transfer Robust Ridge — diff step has fold info for Adaptive
    model_w = fit_robust_center_no_intercept(
        train_X1, train_y1, COMMON_GRID, DELTA_PARAM, ETA_PARAM,
    )
    pred_source_on_target = predict_robust_center_no_intercept(model_w, train_X)
    Y_adjusted = train_y - pred_source_on_target
    model_diff = fit_robust_center_no_intercept(
        train_X, Y_adjusted, COMMON_GRID, DELTA_PARAM, ETA_PARAM, return_folds=True,
    )
    pred_tr = (
        predict_robust_center_no_intercept(model_w, test_X)
        + predict_robust_center_no_intercept(model_diff, test_X)
    )

    # 3) Pooled Robust Ridge
    XX = np.vstack((train_X, train_X1))
    YY = np.concatenate((train_y, train_y1))
    model_pr = fit_robust_center_no_intercept(XX, YY, COMMON_GRID, DELTA_PARAM, ETA_PARAM)
    pred_pr = predict_robust_center_no_intercept(model_pr, test_X)

    # 4) Adaptive Trans-RR (Trans-RR-Ada): convex combination via CV-selected theta
    # Single-RR and Trans-RR target step share the same KFold partition (random_state=1
    # inside find_tau_opt, applied to two arrays of the same length n_train).
    theta_star, cv_curve = adaptive_select_theta_realdata(
        fold_indices=model_sr['fold_indices'],
        fold_betas_sr=model_sr['fold_betas'],
        fold_betas_diff=model_diff['fold_betas'],
        train_X=train_X, train_y=train_y,
        model_w_beta=model_w['beta'],
        mu_train_y1=model_w['y_stats']['mu'],
        mu_Y_adjusted=model_diff['y_stats']['mu'],
        mu_train_y=model_sr['y_stats']['mu'],
        theta_grid=THETA_GRID,
        criterion='mae',
    )
    pred_ada = theta_star * pred_tr + (1 - theta_star) * pred_sr

    # 5) Single Lasso
    lasso_sl = fit_lasso_center_no_intercept(train_X, train_y, COMMON_GRID, split_seed)
    pred_sl = predict_lasso_center_no_intercept(lasso_sl, test_X)

    # 6) Transfer Lasso (two-stage LassoCV)
    lasso_t1 = fit_lasso_center_no_intercept(train_X1, train_y1, COMMON_GRID, split_seed)
    pred_l1_on_target = predict_lasso_center_no_intercept(lasso_t1, train_X)
    Y_offset_adjusted = train_y - pred_l1_on_target
    lasso_t2 = fit_lasso_center_no_intercept(train_X, Y_offset_adjusted, COMMON_GRID, split_seed)
    pred_tl = (
        predict_lasso_center_no_intercept(lasso_t1, test_X)
        + predict_lasso_center_no_intercept(lasso_t2, test_X)
    )

    preds = {
        "single_ridge":       pred_sr,
        "transfer_ridge":     pred_tr,
        "transfer_ridge_ada": pred_ada,
        "pooled_ridge":       pred_pr,
        "single_lasso":       pred_sl,
        "transfer_lasso":     pred_tl,
    }

    rows = []
    for model_name, pred in preds.items():
        mse = float(np.mean((test_y - pred) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(test_y, pred))
        rows.append({
            "direction":       direction_name,
            "repeat_id":       int(split_seed),
            "model":           model_name,
            "MSE":             mse,
            "RMSE":            rmse,
            "R2":              r2,
            "tau_sr":          model_sr["tau"],
            "tau_tr_stage1":   model_w["tau"],
            "tau_tr_stage2":   model_diff["tau"],
            "tau_pr":          model_pr["tau"],
            "alpha_sl":        lasso_sl["alpha"],
            "alpha_tl_stage1": lasso_t1["alpha"],
            "alpha_tl_stage2": lasso_t2["alpha"],
            "ada_theta":       theta_star,
        })
    return rows


def run_repeated_direction(direction_name, target_train, target_test, source_train, n_repeats, seed_start):
    rows = []
    for r in range(n_repeats):
        split_seed = seed_start + r
        rows.extend(run_one_split(direction_name, target_train, target_test, source_train, split_seed))
    return rows


# Run all 5 configurations (main + 3 offsets + no-whiten)

import time

all_results_per_config = {}
for config in CONFIGS:
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"=== Running config: {name}  STEP={config['STEP']}, OFFSET={config['OFFSET']}, WHITEN={config['WHITEN']} ===")
    print(f"{'='*60}")

    Xtt, Xtv, Xst, Xsv, p_used = preprocess(config["STEP"], config["OFFSET"], config["WHITEN"])
    print(f"  predictors p = {p_used}")

    t0 = time.time()
    rows_A = run_repeated_direction(
        direction_name="A_target_is_X",
        target_train=Xtt, target_test=Xtv,
        source_train=Xst,
        n_repeats=N_REPEATS, seed_start=REPEAT_SEED_START,
    )
    rows_B = run_repeated_direction(
        direction_name="B_target_is_X1",
        target_train=Xst, target_test=Xsv,
        source_train=Xtt,
        n_repeats=N_REPEATS, seed_start=REPEAT_SEED_START,
    )
    df = pd.DataFrame(rows_A + rows_B)
    df["config"] = name
    df["p"] = p_used
    df["whiten"] = config["WHITEN"]
    df["offset"] = config["OFFSET"]
    all_results_per_config[name] = df

    print(f"  config '{name}' wall time: {(time.time() - t0)/60:.1f} min")

all_repeat_df = pd.concat(all_results_per_config.values(), ignore_index=True)
print(f"\nTotal rows: {len(all_repeat_df)} (= {len(CONFIGS)} configs x 2 directions x {N_REPEATS} splits x 6 methods)")
all_repeat_df.head()


# Summary: mean and std RMSE per (config, direction, model)
summary_df = (
    all_repeat_df
    .groupby(["config", "direction", "model"], as_index=False)
    .agg(
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        mean_MSE=("MSE", "mean"),
        std_MSE=("MSE", "std"),
        mean_R2=("R2", "mean"),
        std_R2=("R2", "std"),
        mean_ada_theta=("ada_theta", "mean"),
    )
    .sort_values(["config", "direction", "mean_RMSE"])
)
summary_df


# Main config result (this is the row that goes into Table 2)
print("=" * 60)
print("Main config (STEP=4, OFFSET=0, WHITEN=True)")
print("=" * 60)
main_summary = summary_df[summary_df["config"] == "main"]
for direction in main_summary["direction"].unique():
    print(f"\n[{direction}]")
    sub = main_summary[main_summary["direction"] == direction]
    for _, row in sub.iterrows():
        print(f"  {row['model']:22s} | RMSE={row['mean_RMSE']:.4f} +/- {row['std_RMSE']:.4f}  |  R2={row['mean_R2']:.4f}")

# Cross-config robustness pivot (Direction A only for compactness; Direction B is symmetric)
print("\n" + "=" * 60)
print("Robustness across configs (Direction A, mean RMSE)")
print("=" * 60)
pivot_A = (
    summary_df[summary_df["direction"] == "A_target_is_X"]
    .pivot(index="model", columns="config", values="mean_RMSE")
    .reindex(columns=[c["name"] for c in CONFIGS])
)
print(pivot_A.round(4))

print("\n=== Same table for Direction B ===")
pivot_B = (
    summary_df[summary_df["direction"] == "B_target_is_X1"]
    .pivot(index="model", columns="config", values="mean_RMSE")
    .reindex(columns=[c["name"] for c in CONFIGS])
)
print(pivot_B.round(4))


# Diagnostic: distribution of selected theta_star across the 20 splits per config
import collections

print("=== Distribution of Adaptive theta_star (selected weight on Trans-RR) per config ===")
ada_only = all_repeat_df[all_repeat_df["model"] == "transfer_ridge_ada"]
for cfg_name in [c["name"] for c in CONFIGS]:
    for direction in ada_only["direction"].unique():
        sub = ada_only[(ada_only["config"] == cfg_name) & (ada_only["direction"] == direction)]
        thetas = sub["ada_theta"].to_numpy()
        counts = collections.Counter(thetas.round(2))
        c_sorted = ", ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items()))
        print(f"  [{cfg_name:9s} {direction:18s}]  mean={thetas.mean():.3f}  counts: {c_sorted}")


# Save raw results so downstream analysis (boxplots, paper Tables 2/3) does not
# require re-running. We persist three things:
#   (1) Combined long-form CSV (all splits, all configs, all methods) for the master record.
#   (2) Per-config compact JSON with the n_splits x n_methods RMSE matrix per direction
#       (used by the boxplot generator and the LaTeX Table 2/3 rendering).
import os, json
os.makedirs("res", exist_ok=True)

all_repeat_df.to_csv("res/realdata_per_split_all_configs_mae.csv", index=False)
summary_df.to_csv("res/realdata_summary_all_configs_mae.csv", index=False)

method_names = [
    "single_ridge", "transfer_ridge", "transfer_ridge_ada",
    "pooled_ridge", "single_lasso", "transfer_lasso",
]

for cfg in CONFIGS:
    name = cfg["name"]
    df = all_results_per_config[name]
    payload = {
        "config_name": name,
        "STEP": cfg["STEP"],
        "OFFSET": cfg["OFFSET"],
        "WHITEN": cfg["WHITEN"],
        "n_splits": int(N_REPEATS),
        "method_names": method_names,
        "directions": {},
    }
    for direction in df["direction"].unique():
        sub = df[df["direction"] == direction]
        rmse_matrix = (
            sub.pivot(index="repeat_id", columns="model", values="RMSE")
              .reindex(columns=method_names)
              .to_numpy()
              .tolist()
        )
        payload["directions"][direction] = rmse_matrix

    with open(f"res/realdata_rmse_{name}_mae.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: res/realdata_rmse_{name}_mae.json")

print("\nSaved combined:")
print("  res/realdata_per_split_all_configs.csv")
print("  res/realdata_summary_all_configs.csv")

