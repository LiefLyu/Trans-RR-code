"""Generate LaTeX source for the three sensitivity tables that go in Appendix B
of major_v1.tex. Prints to stdout; copy-paste into the .tex file.

Tables produced:
  Table B1 (delta, eta) heatmap     -- Case II Cauchy, dd=1.0, K=200
  Table B2 MSE-CV vs MAE-CV         -- 3 cases x 3 h x 4 methods
  Table B3 Pseudo-Huber vs Huber    -- 3 cases x 3 h x 4 methods
"""
import json
from pathlib import Path

import numpy as np

EXP = Path("/Users/lvlingfeng/Desktop/Trans-ridge/experiment")


def load(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Table B1: (delta, eta) heatmap
# ============================================================
print(r"""% ============================================================
% Table B1: (delta, eta) sensitivity heatmap (Case II, dd = 1.0, K = 200)
% ============================================================
\begin{table}[htbp]
\centering
\setlength\tabcolsep{8pt}
\caption{Sensitivity of relative estimation error to the smoothed Huber loss parameters $(\delta, \eta)$. Setup: Case $\II$ (Cauchy errors), $p = 400$, $n = 400$, $h = 1.0$, $K = 200$ replications. Each cell reports Single-RR / Trans-RR / Trans-RR-Ada mean errors. The variation of Trans-RR across the twelve cells is about $5\%$ of its mean and the ranking is preserved in every cell.}
\label{tab:sens_delta_eta}
\begin{tabular}{c c c c}
\toprule
$\delta \backslash \eta$ & $0.05$ & $0.10$ & $0.20$ \\
\midrule""")

heatmap = load(EXP / "simu_compare/res/2sensitivity_delta_eta_p400_simu200.json")
deltas = heatmap["delta_grid"]
etas = heatmap["eta_grid"]
cells = heatmap["cells"]

for d in deltas:
    row = [f"${d:.2f}$"]
    for e in etas:
        key = f"delta={d}_eta={e}"
        c = cells[key]
        row.append(f"{c['single_mean']:.3f} / {c['trans_mean']:.3f} / {c['adaptive_mean']:.3f}")
    print("    " + " & ".join(row) + r" \\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ============================================================
# Helper for B2 / B3: render a (case x h) x methods table
# ============================================================
CASE_LABEL = {"gaussian": r"$\I$", "cauchy": r"$\II$", "mix": r"$\III$"}


def render_alt_sweep(label, json_path, sweep_key, baseline_path):
    payload = load(json_path)
    cells = payload["sweeps"][sweep_key]
    print(rf"""% ============================================================
% Table B{label['num']}: {label['title']}
% ============================================================
\begin{{table}}[htbp]
\centering
\setlength\tabcolsep{{10pt}}
\caption{{{label['caption']}}}
\label{{tab:sens_{label['key']}}}
\begin{{tabular}}{{c c r r r r}}
\toprule
Case & $h$ & Single-RR & Trans-RR & Trans-RR-Ada & Pooled-RR \\
\midrule""")
    prev_case = None
    for cell in cells:
        case_str = CASE_LABEL[cell["case"]] if cell["case"] != prev_case else ""
        prev_case = cell["case"]
        m = cell["mean_err"]
        # Bold the smallest of SR / TR / Ada / PR for each row
        best_idx = int(np.argmin(m))
        formatted = []
        for i, v in enumerate(m):
            s = f"{v:.4f}"
            if i == best_idx:
                s = r"\textbf{" + s + "}"
            formatted.append(s)
        print(rf"    {case_str} & ${cell['dd']:.3f}$ & " + " & ".join(formatted) + r" \\")
        if cell == cells[-1] or cells[cells.index(cell) + 1]["case"] != cell["case"]:
            if cell != cells[-1]:
                print(r"    \midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Table B2: MSE-CV
# ============================================================
render_alt_sweep(
    label={
        "num": "2",
        "title": "MSE-CV vs MAE-CV (default) sensitivity",
        "key": "mse_cv",
        "caption": (
            "Sensitivity of relative estimation error to the cross-validation criterion. "
            "All settings as in Figure \\ref{figmethods} except the validation loss for ridge-penalty selection is changed from MAE to MSE. "
            "Setup: $K = 500$ replications per cell. Bold marks the smallest mean per row. "
            "Under Case $\\I$ the rankings agree with the MAE-CV main results; under Cases $\\II$ and $\\III$, MSE-CV destabilises Trans-RR, which becomes uniformly worse than Single-RR."
        ),
    },
    json_path=EXP / "simu_compare/res/2sensitivity_cv_loss_p400_simu500.json",
    sweep_key="mse_smoothed_huber",
    baseline_path=None,
)

# ============================================================
# Table B3: Pseudo-Huber
# ============================================================
render_alt_sweep(
    label={
        "num": "3",
        "title": "Pseudo-Huber vs smoothed Huber (default)",
        "key": "pseudo_huber",
        "caption": (
            "Sensitivity of relative estimation error to the choice of robust loss. "
            "All settings as in Figure \\ref{figmethods} except the smoothed Huber loss is replaced by the pseudo-Huber loss "
            "$\\rho_{\\mathrm{PH}}(t; \\delta) = \\delta^2(\\sqrt{1 + (t/\\delta)^2} - 1)$ with $\\delta = 1.35$ (the smoothing parameter $\\eta$ is no longer needed). "
            "Setup: $K = 500$ replications per cell. Bold marks the smallest mean per row. "
            "The rankings agree with the smoothed-Huber main results across all three cases and the discrepancy sweep."
        ),
    },
    json_path=EXP / "simu_compare/res/2sensitivity_cv_loss_p400_simu500.json",
    sweep_key="mae_pseudo_huber",
    baseline_path=None,
)
