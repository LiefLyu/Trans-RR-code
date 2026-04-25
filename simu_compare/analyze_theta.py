"""Analyze the theta diagnostic results: histogram per h + summary table.

Reads:  res/2theta_diagnostic.json
Writes: res/theta_diagnostic_summary.txt
        res/theta_diagnostic_hist.pdf
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino"],
    "font.size": 11,
    "mathtext.fontset": "stix",
})


def main():
    src = Path("res/2theta_diagnostic.json")
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run diagnose_theta.py first.")

    with src.open() as f:
        payload = json.load(f)

    cells = payload["cells"]
    K = payload["K"]

    # ---- Summary table ----
    lines = []
    lines.append(f"{'h (dd)':<10} {'theta_mean':>12} {'theta_med':>10} {'at-1':>6} {'at-0':>6} {'interior':>9} {'SR':>8} {'TR':>8} {'Ada':>8}")
    lines.append("-" * 90)
    for c in cells:
        s = c["summary"]
        lines.append(
            f"{c['dd']:<10.4f} {s['theta_mean']:>12.3f} {s['theta_median']:>10.2f} "
            f"{s['n_at_one']:>6d} {s['n_at_zero']:>6d} {s['n_interior']:>9d} "
            f"{s['single_mean']:>8.4f} {s['trans_mean']:>8.4f} {s['adaptive_mean']:>8.4f}"
        )
    summary = "\n".join(lines)
    print(summary)

    out_txt = Path("res/theta_diagnostic_summary.txt")
    out_txt.write_text(summary + "\n")
    print(f"\nSaved: {out_txt}")

    # ---- Histogram per h ----
    n_cells = len(cells)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), dpi=200)
    axes = axes.flatten()

    bins = np.arange(-0.05, 1.06, 0.1)  # bins centered on grid points {0, 0.1, ..., 1}

    for ax, cell in zip(axes, cells):
        thetas = np.array(cell["theta_star"])
        ax.hist(thetas, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.5)

        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r"$\hat\theta$")
        ax.set_ylabel("count")
        ax.set_ylim(0, K)

        s = cell["summary"]
        title = (
            rf"$h={cell['dd']:.3f}$"
            + "\n"
            + rf"interior:{s['n_interior']}/{K}, $\bar{{\theta}}={s['theta_mean']:.2f}$"
        )
        ax.set_title(title, fontsize=10)

        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, color="#CCC", alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(
        rf"Distribution of $\hat\theta$ at six discrepancy levels (Case I, K={K} reps each)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_pdf = Path("res/theta_diagnostic_hist.pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
