"""Generate methods_comparison_corr.pdf from the AR(1) results.

Reads:  res/2corr_ar1_results_p400_simu1000.json
Writes: res/methods_comparison_corr.pdf  (then to be copied to ustc-overleaf-git/plot/)

Two-panel boxplot, 4 methods, two correlation levels rho ∈ {0.3, 0.6}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino"],
    "font.size": 12,
    "mathtext.fontset": "stix",
})

METHOD_ORDER = ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"]
METHOD_LABELS = {
    "Single RR":    "Single-RR",
    "Trans RR":     "Trans-RR",
    "Trans-RR-Ada": "Trans-RR-Ada",
    "Pooled RR":    "Pooled-RR",
}


def main():
    src = Path("res/2corr_ar1_results_p400_simu1000.json")
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    with src.open() as f:
        data = json.load(f)

    # Group by rho
    rhos_used = sorted({float(item["rho"]) for item in data.values()})
    n_rows = len(rhos_used)

    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 6.5), dpi=300)
    if n_rows == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None

    for ax_idx, rho in enumerate(rhos_used):
        # Collect all cells with this rho
        rows = []
        for key, item in data.items():
            if abs(float(item["rho"]) - rho) > 1e-9:
                continue
            dd_val = float(item["dd"])
            log_dd = np.log(dd_val)
            df_d = item["results_df"]
            for method in METHOD_ORDER:
                if method not in df_d:
                    continue
                for _, err in df_d[method].items():
                    rows.append({
                        "log_dd": log_dd,
                        "method": METHOD_LABELS[method],
                        "error":  err,
                    })
        plot_df = pd.DataFrame(rows)
        plot_df["method"] = pd.Categorical(
            plot_df["method"],
            categories=[METHOD_LABELS[m] for m in METHOD_ORDER],
            ordered=True,
        )

        ax = axes[ax_idx]
        palette = sns.color_palette("muted", n_colors=len(METHOD_ORDER))
        sns.boxplot(
            x="log_dd", y="error", hue="method",
            data=plot_df, palette=palette,
            linewidth=0.8, fliersize=2, ax=ax,
        )

        ax.set_ylabel("Relative Estimation Error", fontsize=12, fontweight="bold")
        if ax_idx == n_rows - 1:
            ax.set_xlabel(r"$\log \|\beta_0-w_0\|$", fontsize=16, fontweight="bold")
        else:
            ax.set_xlabel("")
        log_dd_sorted = sorted(plot_df["log_dd"].unique())
        ax.set_xticks(range(len(log_dd_sorted)))
        ax.set_xticklabels([f"{v:.1f}" for v in log_dd_sorted], rotation=0)

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=2))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 4, 6, 8], numticks=3))

        def fmt(v, _):
            if v >= 1:
                return f"{v:g}"
            return f"{v:.3f}".rstrip("0").rstrip(".")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_minor_formatter(FuncFormatter(fmt))
        ax.tick_params(axis="y", labelsize=12)

        ax.text(
            0.5, 0.9, rf"$\rho = {rho}$", transform=ax.transAxes,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#666666"),
            fontsize=14, fontweight="bold",
        )
        ax.set_title("")

        ax.grid(True, linestyle="--", linewidth=0.8, color="#CCCCCC", alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#444444")
        ax.tick_params(width=0.8)

        if ax_idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_labels),
        framealpha=0.9, edgecolor="black", fancybox=True, fontsize=11,
    )
    fig.patch.set_facecolor("white")

    out = Path("res/methods_comparison_corr.pdf")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
