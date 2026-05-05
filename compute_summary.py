"""Compute all the summary numbers needed to fill the [TODO] placeholders in
ustc-overleaf-git/major_v1.tex. Reads everything from res/*.json across the
experiment subdirectories and prints a structured report.

Sections of the report match the [TODO] sites in the LaTeX:
- §4.3 Adaptive narrative (h transition behavior)
- §4.4 (delta, eta) heatmap variation
- §4.4 MSE-CV vs MAE-CV ranking
- §4.4 pseudo-Huber vs smoothed-Huber
- §5 Table 2 (realdata main config)
- §5 Robustness (offset variants + no-whitening + MAE comparison)
"""
import json
from pathlib import Path

import numpy as np

EXP = Path("/Users/lvlingfeng/Desktop/Trans-ridge/experiment")


def safe_load(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "N/A"


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================================
section("§4.3 Adaptive Trans-RR narrative (Gaussian Case I)")
gauss = safe_load(EXP / "simu_compare/res/2gaussian_cv_results_p400_simu500.json")
if gauss is None:
    print("  NO DATA: 2gaussian_cv_results_p400_simu500.json not yet present")
else:
    print(f"  {'h (dd)':<10} {'SR':>8} {'TR':>8} {'Ada':>8} {'PR':>8} {'SL':>8} {'TL':>8}")
    for dd_str in sorted(gauss.keys(), key=float):
        cell = gauss[dd_str]
        m = cell["mean_err"]
        # mean_err order: [Single RR, Trans RR, Trans-RR-Ada, Pooled RR, Single Lasso, Trans Lasso]
        print(f"  {float(dd_str):<10.4f} {m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>8.4f} {m[3]:>8.4f} {m[4]:>8.4f} {m[5]:>8.4f}")
    # Find dd closest to 1.0 for "transition" comment
    dds = [float(k) for k in gauss.keys()]
    transition_dd = min(dds, key=lambda x: abs(x - 1.0))
    cell_t = gauss[str(transition_dd)] if str(transition_dd) in gauss else gauss[next(iter(gauss))]
    m = cell_t["mean_err"]
    print()
    print(f"  Transition h (closest to 1.0) = {transition_dd:.4f}")
    print(f"    Single-RR={m[0]:.4f}  Trans-RR={m[1]:.4f}  Trans-RR-Ada={m[2]:.4f}")
    if m[2] <= min(m[0], m[1]):
        print(f"    -> Adaptive at-or-better than both bases at the transition")

# ============================================================================
section("§4.3 + post_hoc theta_star distribution from main JSON")
# Theta is now extracted from the main experiment JSON's 'theta_star' field
# (saved per replicate). The standalone diagnose_theta.py is obsolete; use
# post_hoc_theta_summary.py instead, or read directly here.
gauss_post = safe_load(EXP / "simu_compare/res/2gaussian_cv_results_p400_simu500.json")
if gauss_post is None or "theta_star" not in next(iter(gauss_post.values()), {}):
    print("  NO DATA (or main JSON pre-dates the schema with 'theta_star' field)")
else:
    print(f"  {'h':<10} {'theta_mean':>12} {'theta_med':>10} {'at-1 %':>8} {'at-0 %':>8} {'interior %':>11}")
    for dd_str in sorted(gauss_post.keys(), key=float):
        cell = gauss_post[dd_str]
        theta = np.asarray(cell.get("theta_star", []), dtype=float)
        if len(theta) == 0:
            continue
        eps = 1e-6
        n_one  = int((theta >= 1 - eps).sum())
        n_zero = int((theta <=     eps).sum())
        n_int  = int(((theta > eps) & (theta < 1 - eps)).sum())
        K = len(theta)
        print(f"  {float(dd_str):<10.4f} {theta.mean():>12.3f} {np.median(theta):>10.2f} "
              f"{100*n_one/K:>7.1f}% {100*n_zero/K:>7.1f}% {100*n_int/K:>10.1f}%")
    # Interior fraction at h~1
    dds = [float(k) for k in gauss_post.keys()]
    h1 = min(dds, key=lambda x: abs(x - 1.0))
    cell_h1 = gauss_post[str(h1)]
    theta_h1 = np.asarray(cell_h1.get("theta_star", []), dtype=float)
    if len(theta_h1) > 0:
        n_int_h1 = int(((theta_h1 > 1e-6) & (theta_h1 < 1 - 1e-6)).sum())
        K_h1 = len(theta_h1)
        print(f"\n  -> §4.3 Adaptive narrative: at h={h1:.4f}, interior theta in {n_int_h1}/{K_h1} = {100*n_int_h1/K_h1:.0f}% of replications")

# ============================================================================
section("§4.5 (delta, eta) heatmap sensitivity (B1)")
heatmap = safe_load(EXP / "simu_compare/res/2sensitivity_delta_eta_p400_simu500.json")
if heatmap is None:
    print("  NO DATA")
else:
    # Schema: cells is list of dicts with case, h, delta, eta, mean_err (4 methods)
    cells = heatmap["cells"]
    print(f"  cells: {len(cells)}")
    # Aggregate Trans-RR variation across (delta, eta) per (case, h)
    print(f"\n  {'case':<10} {'h':<10} {'TR min':>10} {'TR max':>10} {'TR range %':>12}")
    by_ch = {}
    for c in cells:
        key = (c["case"], round(c["h"], 4))
        by_ch.setdefault(key, []).append(c["mean_err"])
    for (case, h), arrs in sorted(by_ch.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        arrs = np.array(arrs)            # (9 cells, 4 methods)
        tr_means = arrs[:, 1]            # Trans-RR column
        rng_pct = 100 * (tr_means.max() - tr_means.min()) / tr_means.mean()
        print(f"  {case:<10} {h:<10.4f} {tr_means.min():>10.4f} {tr_means.max():>10.4f} {rng_pct:>11.1f}%")
    # Ranking-preservation check across all (case, h) cells
    n_preserved = 0
    n_total = 0
    for c in cells:
        m = c["mean_err"]
        # ranking: order of (Single, Trans, Ada, Pooled)
        # Just record whether ada is at-or-better than min(Single, Trans)
        n_total += 1
        if m[2] <= min(m[0], m[1]) + 1e-9:
            n_preserved += 1
    print(f"\n  Ada at-or-better than better base in {n_preserved}/{n_total} cells")

# ============================================================================
section("§4.4 MSE-CV / pseudo-Huber sensitivity")
sens = safe_load(EXP / "simu_compare/res/2sensitivity_cv_loss_p400_simu500.json")
if sens is None:
    print("  NO DATA")
else:
    for sweep_name, cells in sens["sweeps"].items():
        print(f"\n  Sweep: {sweep_name}")
        print(f"    {'case':<10} {'h':<10} {'SR':>8} {'TR':>8} {'Ada':>8} {'PR':>8}")
        for c in cells:
            m = c["mean_err"]
            print(f"    {c['case']:<10} {c['dd']:<10.4f} {m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>8.4f} {m[3]:>8.4f}")
    print()
    # Compare main vs MSE-CV vs pseudo-Huber for ranking preservation:
    # We say "ranking preserved" if Trans-RR < Single-RR for small h in all sweeps.

# ============================================================================
section("§5 Table 2: realdata main config (MSE-CV)")
real_main = safe_load(EXP / "realdata/res/realdata_rmse_main.json")
if real_main is None:
    print("  NO DATA")
else:
    method_names = real_main["method_names"]
    n_splits = real_main["n_splits"]
    for direction in real_main["directions"]:
        rmse = np.array(real_main["directions"][direction])
        mean = rmse.mean(axis=0)
        std = rmse.std(axis=0)
        print(f"\n  {direction}:")
        print(f"    {'method':<22} {'mean':>10} {'std':>10}")
        for i, m in enumerate(method_names):
            print(f"    {m:<22} {mean[i]:>10.4f} {std[i]:>10.4f}")

# ============================================================================
section("§5 Table 3: realdata robustness across configs (MSE-CV)")
configs = ["main", "off1", "off2", "off3", "no_whiten"]
real_data = {}
for c in configs:
    real_data[c] = safe_load(EXP / f"realdata/res/realdata_rmse_{c}.json")

if all(v is None for v in real_data.values()):
    print("  NO DATA")
else:
    for direction in ["A_target_is_X", "B_target_is_X1"]:
        print(f"\n  Direction {direction}, mean RMSE per config:")
        method_names = next((v["method_names"] for v in real_data.values() if v is not None), None)
        if method_names is None:
            continue
        # Header
        header = f"    {'method':<22}"
        for c in configs:
            header += f" {c:>10}"
        print(header)
        for i, m in enumerate(method_names):
            row = f"    {m:<22}"
            for c in configs:
                v = real_data[c]
                if v is None or direction not in v["directions"]:
                    row += f" {'N/A':>10}"
                else:
                    rmse = np.array(v["directions"][direction])
                    row += f" {rmse.mean(axis=0)[i]:>10.4f}"
            print(row)

# ============================================================================
section("§5 Phase 3a decision: realdata MAE vs MSE comparison (main config)")
mse_main = safe_load(EXP / "realdata/res/realdata_rmse_main.json")
mae_main = safe_load(EXP / "realdata/res/realdata_rmse_main_mae.json")
if mse_main is None or mae_main is None:
    print("  NO DATA (either MSE or MAE realdata not yet done)")
else:
    print(f"  {'method':<22} {'MSE-mean A':>14} {'MAE-mean A':>14} {'MSE-mean B':>14} {'MAE-mean B':>14}")
    method_names = mse_main["method_names"]
    for i, m in enumerate(method_names):
        rs_mse_A = np.array(mse_main["directions"]["A_target_is_X"])[:, i].mean()
        rs_mae_A = np.array(mae_main["directions"]["A_target_is_X"])[:, i].mean()
        rs_mse_B = np.array(mse_main["directions"]["B_target_is_X1"])[:, i].mean()
        rs_mae_B = np.array(mae_main["directions"]["B_target_is_X1"])[:, i].mean()
        print(f"  {m:<22} {rs_mse_A:>14.4f} {rs_mae_A:>14.4f} {rs_mse_B:>14.4f} {rs_mae_B:>14.4f}")
    # Determine if ranking preserved
    rmse_mse_A = np.array(mse_main["directions"]["A_target_is_X"]).mean(axis=0)
    rmse_mae_A = np.array(mae_main["directions"]["A_target_is_X"]).mean(axis=0)
    order_mse = np.argsort(rmse_mse_A)
    order_mae = np.argsort(rmse_mae_A)
    same = np.array_equal(order_mse, order_mae)
    print(f"\n  Direction A method ranking preserved between MSE and MAE: {same}")
    if same:
        print(f"  -> Recommend MAE for §5 (consistency with §4)")
    else:
        print(f"  -> Differ; investigate before deciding")
