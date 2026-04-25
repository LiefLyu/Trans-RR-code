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
gauss = safe_load(EXP / "simu_compare/res/2gaussian_cv_results_p400_simu1000.json")
if gauss is None:
    print("  NO DATA: 2gaussian_cv_results_p400_simu1000.json not yet present")
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
section("§4.3 + diagnose_theta theta_star distribution")
diag = safe_load(EXP / "simu_compare/res/2theta_diagnostic.json")
if diag is None:
    print("  NO DATA")
else:
    print(f"  {'h':<10} {'theta_mean':>12} {'theta_med':>10} {'at-1':>6} {'at-0':>6} {'interior':>9}")
    for c in diag["cells"]:
        s = c["summary"]
        print(f"  {c['dd']:<10.4f} {s['theta_mean']:>12.3f} {s['theta_median']:>10.2f} "
              f"{s['n_at_one']:>6d} {s['n_at_zero']:>6d} {s['n_interior']:>9d}")
    # Interior fraction at h~1
    h1_cell = next((c for c in diag["cells"] if abs(c["dd"] - 1.0) < 1e-6), None)
    if h1_cell:
        n_int = h1_cell["summary"]["n_interior"]
        K = h1_cell["summary"]["K"]
        print(f"\n  -> §4.3 Adaptive narrative number: interior theta in {n_int}/{K} = {100*n_int/K:.0f}% of replications at h=1.0")

# ============================================================================
section("§4.4 (delta, eta) sensitivity heatmap")
heatmap = safe_load(EXP / "simu_compare/res/2sensitivity_delta_eta_p400_simu200.json")
if heatmap is None:
    print("  NO DATA")
else:
    cells = heatmap["cells"]
    print(f"  {'(delta, eta)':<16} {'Trans-RR mean':>14} {'Single-RR mean':>16} {'Adaptive mean':>15}")
    trans_means = []
    for k, v in cells.items():
        d, e = v["delta"], v["eta"]
        tr = v["trans_mean"]; sr = v["single_mean"]; ad = v["adaptive_mean"]
        trans_means.append(tr)
        print(f"  ({d:.2f}, {e:.2f})    {tr:>14.4f} {sr:>16.4f} {ad:>15.4f}")
    rng_pct = 100 * (max(trans_means) - min(trans_means)) / np.mean(trans_means)
    print(f"\n  Trans-RR variation across 12 cells: max={max(trans_means):.4f}, min={min(trans_means):.4f}, range={rng_pct:.1f}% of mean")
    print(f"  -> §4.4 (delta, eta) paragraph fill: '... varies by {rng_pct:.1f}\\%'")

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
