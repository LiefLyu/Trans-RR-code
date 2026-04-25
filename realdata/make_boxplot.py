"""Generate the realdata boxplot from the main config's saved JSON.

Reviewer 1 Minor 2: add boxplot of RMSE across the 20 splits.
Reviewer 2 Comment 5 (b): report distributional information beyond mean/SD.

Reads:  res/realdata_rmse_main.json
Writes: res/realdata_boxplot.pdf  (then copy to ustc-overleaf-git/plot/)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino"],
    "font.size": 12,
    "mathtext.fontset": "stix",
})

# ---- Method label mapping (display name + paper-style hyphenation) ----
DISPLAY = {
    "single_ridge":       "Single-RR",
    "transfer_ridge":     "Trans-RR",
    "transfer_ridge_ada": "Trans-RR-Ada",
    "pooled_ridge":       "Pooled-RR",
    "single_lasso":       "Single-Lasso",
    "transfer_lasso":     "Trans-Lasso",
}

# Order: ridge methods first, then lasso (matches §4 figure)
ORDER = ["single_ridge", "transfer_ridge", "transfer_ridge_ada", "pooled_ridge", "single_lasso", "transfer_lasso"]

DIRECTION_LABEL = {
    "A_target_is_X":   r"Direction A: target = $X$",
    "B_target_is_X1":  r"Direction B: target = $X_1$",
}


def main():
    src = Path("res/realdata_rmse_main.json")
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run realdata 5-config first.")

    with src.open() as f:
        payload = json.load(f)

    n_splits = payload["n_splits"]
    method_names = payload["method_names"]  # original (non-displayed) order
    cols_in_order = [m for m in ORDER if m in method_names]
    display_labels = [DISPLAY[m] for m in cols_in_order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300, sharey=True)
    palette = plt.cm.tab10(np.linspace(0, 1, len(cols_in_order)))

    for ax, direction in zip(axes, ["A_target_is_X", "B_target_is_X1"]):
        rmse_matrix = np.array(payload["directions"][direction])  # n_splits x n_methods (in method_names order)
        # Reorder columns according to ORDER
        col_idx = [method_names.index(m) for m in cols_in_order]
        data = rmse_matrix[:, col_idx]

        bp = ax.boxplot(
            [data[:, j] for j in range(data.shape[1])],
            tick_labels=display_labels,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.4},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "gray", "markeredgecolor": "none"},
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("#444")

        ax.set_title(DIRECTION_LABEL[direction], fontsize=13, fontweight="bold")
        ax.set_ylabel("RMSE" if direction == "A_target_is_X" else "")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color="#CCC", alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#444")

    fig.suptitle(
        f"NIR Shootout: distribution of RMSE over {n_splits} repeated splits",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout()
    out = Path("res/realdata_boxplot.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
