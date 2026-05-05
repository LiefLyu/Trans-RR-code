"""Post-hoc summary of theta_star and h=1 method averages.

Reads the §4.3 main JSON outputs (saved with the new schema in run_*_main.py)
and computes the narrative numbers used in the manuscript:
  - Interior-theta proportion: fraction of replications with theta_star in (0, 1)
  - h=1 average errors per method (Single-RR, Trans-RR, Trans-RR-Ada)

This replaces the standalone diagnose_theta.py.

Usage:
    python post_hoc_theta_summary.py
"""
import json
import os
from pathlib import Path

import numpy as np


CASES_FILES = {
    "Case I (Gaussian)":  "res/2gaussian_cv_results_p400_simu500.json",
    "Case II (Cauchy)":   "res/2cauchy_cv_results_p400_simu500.json",
    "Case III (Mixture)": "res/2mix_cv_results_p400_simu500.json",
}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_h1_key(payload, target_h=1.0):
    """The dd-keys are stringified floats. Find the one closest to target_h."""
    keys = list(payload.keys())
    floats = [float(k) for k in keys]
    idx = int(np.argmin(np.abs(np.array(floats) - target_h)))
    return keys[idx], floats[idx]


def summarize_one(payload, target_h=1.0, eps=1e-6):
    h_key, h_val = find_h1_key(payload, target_h)
    cell = payload[h_key]

    theta = np.asarray(cell["theta_star"], dtype=float)
    interior_pct = float(np.mean((theta > eps) & (theta < 1 - eps))) * 100
    at_zero_pct  = float(np.mean(theta <= eps)) * 100
    at_one_pct   = float(np.mean(theta >= 1 - eps)) * 100

    mean_err = cell["mean_err"]
    method_names = ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"]
    if len(mean_err) == 6:
        method_names = method_names + ["Single Lasso", "Trans Lasso"]

    return {
        "h_key": h_key, "h_val": h_val,
        "interior_pct": interior_pct,
        "at_zero_pct":  at_zero_pct,
        "at_one_pct":   at_one_pct,
        "method_names": method_names,
        "mean_err":     mean_err,
    }


def main():
    print("=== Post-hoc theta summary @ h ≈ 1.0 ===")
    print()
    for case_name, path in CASES_FILES.items():
        full_path = os.path.join(_THIS_DIR, path)
        if not Path(full_path).exists():
            print(f"  [skip] {case_name}: {path} not found")
            continue
        with open(full_path) as f:
            payload = json.load(f)
        s = summarize_one(payload, target_h=1.0)
        print(f"{case_name}  h={s['h_val']:.4f} (key={s['h_key']!r})")
        print(f"  theta interior (0,1): {s['interior_pct']:.1f}%   "
              f"theta=0: {s['at_zero_pct']:.1f}%   theta=1: {s['at_one_pct']:.1f}%")
        names = s['method_names']
        for name, err in zip(names, s['mean_err']):
            print(f"    {name:14}: {err:.4f}")
        print()


if __name__ == "__main__":
    main()
