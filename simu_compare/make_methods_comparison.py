"""Generate methods_comparison.pdf from the post-Adaptive 2*_cv_results JSONs.

Reads:  res/2{gaussian,cauchy,mix}_cv_results_p400_simu500.json
Writes: res/methods_comparison.pdf  (then to be copied to ustc-overleaf-git/plot/)

Three-panel boxplot, 6 methods for Case I (Gaussian) and 4 methods for Cases II/III.
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
bg_color = "white"

distribution_types = ["gaussian", "cauchy", "mix"]
file_paths = [
    "res/2gaussian_cv_results_p400_simu500.json",
    "res/2cauchy_cv_results_p400_simu500.json",
    "res/2mix_cv_results_p400_simu500.json",
]

RIDGE_ORDER = ["Single RR", "Trans RR", "Trans-RR-Ada", "Pooled RR"]
LASSO_ORDER = ["Single Lasso", "Trans Lasso"]
METHOD_LABELS = {
    "Single RR":     "Single-RR",
    "Trans RR":      "Trans-RR",
    "Trans-RR-Ada":  "Trans-RR-Ada",
    "Pooled RR":     "Pooled-RR",
    "Single Lasso":  "Single-Lasso",
    "Trans Lasso":   "Trans-Lasso",
}


def main():
    # Verify all 3 JSONs exist
    for fp in file_paths:
        if not Path(fp).exists():
            raise SystemExit(f"Missing input: {fp}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=300)
    legend_handles_full, legend_labels_full = None, None

    for idx, (dist_type, file_path) in enumerate(zip(distribution_types, file_paths)):
        with open(file_path, "r") as f:
            data = json.load(f)

        sample_dd = next(iter(data))
        methods_in_case = list(data[sample_dd]["results_df"].keys())
        case_order = [m for m in RIDGE_ORDER + LASSO_ORDER if m in methods_in_case]

        rows = []
        for dd_str in data:
            dd_val = float(dd_str)
            log_dd = np.log(dd_val)
            df_d = data[dd_str]["results_df"]
            for method in case_order:
                for _, err in df_d[method].items():
                    rows.append({
                        "log_dd": log_dd,
                        "method": METHOD_LABELS[method],
                        "error":  err,
                    })
        plot_df = pd.DataFrame(rows)
        plot_df["method"] = pd.Categorical(
            plot_df["method"],
            categories=[METHOD_LABELS[m] for m in case_order],
            ordered=True,
        )

        ax = axes[idx]
        ax.set_facecolor(bg_color)

        palette = sns.color_palette("muted", n_colors=len(case_order))
        sns.boxplot(
            x="log_dd", y="error", hue="method",
            data=plot_df, palette=palette,
            linewidth=0.8, fliersize=2, ax=ax,
        )

        ax.set_ylabel("Relative Estimation Error", fontsize=12, fontweight="bold")
        if idx == 2:
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

        case_label = {"gaussian": "case I", "cauchy": "case II", "mix": "case III"}[dist_type]
        ax.text(
            0.5, 0.9, case_label, transform=ax.transAxes,
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

        if idx == 0:
            legend_handles_full, legend_labels_full = ax.get_legend_handles_labels()
        ax.legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    fig.legend(
        legend_handles_full, legend_labels_full,
        loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_labels_full),
        framealpha=0.9, edgecolor="black", fancybox=True, fontsize=11,
    )
    fig.patch.set_facecolor("white")

    out = Path("res/methods_comparison.pdf")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
