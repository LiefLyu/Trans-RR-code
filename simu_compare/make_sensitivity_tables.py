"""Generate LaTeX source for the four sensitivity tables in Appendix B.

Tables produced (printed to stdout):
    Tab B1  -- (delta, eta) corner sensitivity, 3 cases x 7 h x 5 (delta, eta) values
               (4 corners + default (1.35, 0.1) merged from main JSON).
    Tab B2  -- MSE-CV vs MAE-CV (default), 3 cases x 7 h x 4 methods.
    Tab B3  -- pseudo-Huber vs smoothed Huber (default), 3 cases x 7 h x 4 methods.
    Tab B4  -- ridge-grid-width sensitivity, default vs wide grid, 3 cases x 7 h x 4 methods.

Inputs:
    res/2gaussian_cv_results_p400_simu500.json   (default (1.35, 0.1) for Tab B1)
    res/2cauchy_cv_results_p400_simu500.json
    res/2mix_cv_results_p400_simu500.json
    res/2sensitivity_delta_eta_p400_simu500.json (corners for Tab B1)
    res/2sensitivity_cv_loss_p400_simu500.json   (Tabs B2 + B3)
    res/2sensitivity_tau_p400_simu500.json       (Tab B4)
"""
import json
from pathlib import Path

import numpy as np

EXP = Path("/Users/lvlingfeng/Desktop/Trans-ridge/experiment")
RES = EXP / "simu_compare" / "res"
M_SENS = 500


def load(path):
    with open(path) as f:
        return json.load(f)


CASE_LABEL = {"gaussian": r"$\I$", "cauchy": r"$\II$", "mix": r"$\III$"}
DEFAULT_DELTA, DEFAULT_ETA = 1.35, 0.1


def find_main_dd_key(payload, target_dd):
    keys = list(payload.keys())
    floats = np.array([float(k) for k in keys])
    idx = int(np.argmin(np.abs(floats - target_dd)))
    return keys[idx]


def main_mean_at_h(case, dd, n_methods=4):
    """Read main JSON for `case` and return (Single, Trans, Ada, Pooled) means at the
    h-cell closest to dd. Drops Lasso columns from Gaussian if present."""
    fname = f"2{case}_cv_results_p400_simu{M_SENS}.json"
    payload = load(RES / fname)
    key = find_main_dd_key(payload, dd)
    mean = payload[key]["mean_err"]
    return mean[:n_methods]


# ============================================================
# Tab B1: (delta, eta) corner sensitivity
# ============================================================
def render_tab_B1(h_target=1.0):
    """Render Tab B1 as three case-specific 3x3 (delta, eta) heatmaps at a
    single representative h value (default h = 1.0, the negative-transfer
    transition). Each cell reports Single-RR / Trans-RR / Trans-RR-Ada /
    Pooled-RR mean error. The three subtables are stacked vertically inside
    one table environment so that the caption applies to all three.
    """
    sens = load(RES / f"2sensitivity_delta_eta_p400_simu{M_SENS}.json")
    cases = sens["cases"]
    deltas = sens["deltas"]      # [1.0, 1.35, 2.0]
    etas   = sens["etas"]        # [0.05, 0.1, 0.2]

    by_key = {}
    for c in sens["cells"]:
        if abs(c["h"] - h_target) < 1e-3:
            by_key[(c["case"], round(c["delta"], 4),
                    round(c["eta"], 4))] = c["mean_err"]

    case_descr = {
        "gaussian": r"Case $\I$ (Gaussian errors)",
        "cauchy":   r"Case $\II$ (Cauchy errors)",
        "mix":      r"Case $\III$ (mixture errors)",
    }

    print(r"""% ============================================================
% Tab B1: (delta, eta) heatmap sensitivity at h = """ + f"{h_target:g}" + r""" (transition).
%   Three case-specific 3x3 subtables stacked vertically inside one table.
%   Cell entry: Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR mean error.
% ============================================================
\begin{table}[htbp]
\centering
\small
\caption{Sensitivity of relative estimation error to the smoothed Huber parameters $(\delta, \eta)$ at the transition discrepancy $h = """ + f"{h_target:g}" + r"""$. Each entry reports Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR mean estimation error over $M = """ + f"{M_SENS}" + r"""$ replications. The three blocks correspond to the three error distributions of Section~\ref{sec:sim_methods}.}
\label{tab:sens_delta_eta}""")
    for ci, case in enumerate(cases):
        if ci > 0:
            print(r"\medskip\par")
        print(rf"\textbf{{{case_descr[case]}}}\\[2pt]")
        print(r"\begin{tabularx}{\textwidth}{c *{3}{>{\centering\arraybackslash}X}}")
        print(r"\toprule")
        print(r"$\delta \backslash \eta$ & $0.05$ & $0.10$ & $0.20$ \\")
        print(r"\midrule")
        for delta in deltas:
            row = [f"${delta:.2f}$"]
            for eta in etas:
                key = (case, round(delta, 4), round(eta, 4))
                if key in by_key:
                    mean = by_key[key]
                    row.append(" / ".join(f"{x:.3f}" for x in mean))
                else:
                    row.append("--")
            print("    " + " & ".join(row) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabularx}")
    print(r"\end{table}")
    print()


# ============================================================
# Tab B2 / B3: alt CV criterion / alt loss, paired with default
# ============================================================
def render_alt_sweep_paired(payload, sweep_key, label_num, table_key, caption):
    """Render Tab B2 or B3 as a paired-comparison table: each cell entry is
    'default / perturbed' so the reader can compare the two side by side.
    Default means come from the main experiment JSONs."""
    cells = payload["sweeps"][sweep_key]
    print(rf"""% ============================================================
% Tab B{label_num}: {table_key} (paired with default for direct comparison)
% ============================================================
\begin{{table}}[htbp]
\centering
\small
\caption{{{caption}}}
\label{{tab:sens_{table_key}}}
\begin{{tabularx}}{{\textwidth}}{{c c *{{4}}{{>{{\centering\arraybackslash}}X}}}}
\toprule
Case & $h$ & Single-RR & Trans-RR & Trans-RR-Ada & Pooled-RR \\
\midrule""")
    cases_in_order = ["gaussian", "cauchy", "mix"]
    for ci, case in enumerate(cases_in_order):
        case_cells = [c for c in cells if c["case"] == case]
        for j, c in enumerate(case_cells):
            case_str = CASE_LABEL[case] if j == 0 else ""
            default_means = main_mean_at_h(case, c["dd"], n_methods=4)
            alt_means = c["mean_err"]
            entries = [f"{d:.3f} / {a:.3f}" for d, a in zip(default_means, alt_means)]
            print(rf"    {case_str} & ${c['dd']:.3f}$ & " + " & ".join(entries) + r" \\")
        if ci < len(cases_in_order) - 1:
            print(r"    \midrule")
    print(r"""\bottomrule
\end{tabularx}
\end{table}
""")


def render_tab_B2_B3():
    sens = load(RES / f"2sensitivity_cv_loss_p400_simu{M_SENS}.json")

    render_alt_sweep_paired(
        sens, sweep_key="mse_smoothed_huber",
        label_num=2, table_key="mse_cv",
        caption=(
            "Sensitivity of relative estimation error to the cross-validation criterion. "
            "All settings as in Figure~\\ref{figmethods} except every cross-validation "
            "loss (used to select $\\tau_1$, $\\tau$, $\\tau_{\\mathrm{st}}$, $\\tau_{\\mathrm{p}}$, "
            "and $\\theta$) is changed from MAE to MSE. "
            "Each entry reports the default (MAE-CV) and the MSE-CV mean estimation error, "
            "in the format ``default $/$ MSE-CV'', over $M = 500$ replications."
        ),
    )

    render_alt_sweep_paired(
        sens, sweep_key="mae_pseudo_huber",
        label_num=3, table_key="pseudo_huber",
        caption=(
            "Sensitivity of relative estimation error to the choice of robust loss. "
            "All settings as in Figure~\\ref{figmethods} except the smoothed Huber loss is "
            "replaced by the pseudo-Huber loss "
            "$\\rho_{\\mathrm{PH}}(t; \\delta) = \\delta^2(\\sqrt{1 + (t/\\delta)^2} - 1)$ with "
            "$\\delta = 1.35$ (the smoothing parameter $\\eta$ is no longer needed). "
            "Each entry reports the default (smoothed Huber) and the pseudo-Huber mean "
            "estimation error, in the format ``default $/$ pseudo-Huber'', "
            "over $M = 500$ replications."
        ),
    )


# ============================================================
# Tab B4: ridge grid-width sensitivity (default vs wide)
# ============================================================
def render_tab_B4():
    """Render Tab B4. Default-grid column comes from the main JSONs (which are
    identical to a re-run on the default grid because ridge is deterministic);
    wide-grid column comes from the dedicated B4 sensitivity JSON."""
    sens = load(RES / f"2sensitivity_tau_p400_simu{M_SENS}.json")
    cases = sens["cases"]
    dd_values = sens["dd_values"]
    cells = sens["cells"]
    # B4 sensitivity JSON only stores the wide-grid run.
    wide_by_key = {(c["case"], round(c["dd"], 4)): c["mean_err"] for c in cells}

    print(r"""% ============================================================
% Tab B4: ridge-grid-width sensitivity
%   default grid: logspace(-2, 2, 9, base=3) -- 9 points in [1/9, 9].
%   wide    grid: logspace(-3, 3, 13, base=3) -- 13 points in [1/27, 27], superset.
%   Default-grid column reuses the main experiment (Figure 3) since the ridge
%   methods are deterministic given (data, tau_grid).
%   Each cell reports Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR mean error.
% ============================================================
\begin{table}[htbp]
\centering
\setlength\tabcolsep{6pt}
\caption{Sensitivity of relative estimation error to the ridge-penalty cross-validation grid. The default grid contains $9$ values from $1/9$ to $9$ on a geometric scale. The wide grid extends this to $13$ values from $1/27$ to $27$ on the same geometric scale, and contains the default grid as a strict subset. All other settings as in Figure~\ref{figmethods}, with $M = 500$ replications per cell. Each entry reports Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR mean estimation error.}
\label{tab:sens_tau}
\small
\begin{tabular}{c c c c}
\toprule
Case & $h$ & default grid (9 pts, $[1/9, 9]$) & wide grid (13 pts, $[1/27, 27]$) \\
\midrule""")
    for case in cases:
        for dd in dd_values:
            row = [CASE_LABEL[case], f"${dd:.3f}$"]
            # Default grid: read from main JSON
            mean_default = main_mean_at_h(case, dd, n_methods=4)
            row.append(" / ".join(f"{x:.3f}" for x in mean_default))
            # Wide grid: read from sensitivity_tau JSON
            key = (case, round(dd, 4))
            if key in wide_by_key:
                m = wide_by_key[key]
                row.append(" / ".join(f"{x:.3f}" for x in m))
            else:
                row.append("--")
            print("    " + " & ".join(row) + r" \\")
        if case != cases[-1]:
            print(r"    \midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ============================================================
# Tab B5: fixed-tau ablation
# ============================================================
def render_tab_B5():
    sens = load(RES / f"2sensitivity_fixed_tau_p400_simu{M_SENS}.json")
    cases = sens["cases"]
    dd_values = sens["dd_values"]
    tau_values = sens["tau_fixed_values"]
    cells = sens["cells"]
    by_key = {(c["case"], round(c["dd"], 4), round(c["tau_fixed"], 4)): c for c in cells}

    # Render: rows = (case, h), columns = each fixed tau (one column per tau,
    # cell entry = Single / Trans / Ada / Pooled mean error).
    tau_headers = []
    for tau in tau_values:
        if abs(tau - 1.0/3.0) < 1e-3:
            tau_headers.append(r"$\tau = 1/3$")
        else:
            tau_headers.append(rf"$\tau = {tau:g}$")

    print(r"""% ============================================================
% Tab B5: Fixed-tau ablation, 3 cases x 7 h x 4 fixed tau values.
%   No tau-CV; all four ridge penalties forced to a common fixed value.
%   Adaptive's theta is still selected by CV on the target sample.
%   Each entry reports Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR mean error.
% ============================================================
\begin{table}[htbp]
\centering
\scriptsize
\setlength\tabcolsep{2pt}
\caption{Sensitivity of relative estimation error to the ridge penalty value when cross-validation tuning is disabled. All four ridge penalties (Single-RR's $\tau_{\mathrm{st}}$, Trans-RR's $\tau_1$ and $\tau$, Pooled-RR's $\tau_{\mathrm{p}}$) are forced to a common fixed value from $\{1/3, 1, 3, 9\}$. Trans-RR-Ada's mixing weight $\theta$ is still selected by 5-fold cross-validation on the target sample (Algorithm~\ref{algo2}). All other settings as in Figure~\ref{figmethods}, with $M = 500$ replications per cell. Each entry reports Single-RR / Trans-RR / Trans-RR-Ada / Pooled-RR mean estimation error.}
\label{tab:sens_fixed_tau}
\begin{tabular}{c c """ + " ".join(["c"] * len(tau_values)) + r"""}
\toprule
Case & $h$ & """ + " & ".join(tau_headers) + r""" \\
\midrule""")
    for case in cases:
        for dd in dd_values:
            row = [CASE_LABEL[case], f"${dd:.3f}$"]
            for tau in tau_values:
                key = (case, round(dd, 4), round(tau, 4))
                if key in by_key:
                    m = by_key[key]["mean_err"]
                    row.append(" / ".join(f"{x:.3f}" for x in m))
                else:
                    row.append("--")
            print("    " + " & ".join(row) + r" \\")
        if case != cases[-1]:
            print(r"    \midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    print(r"% Auto-generated by make_sensitivity_tables.py. Do not edit by hand.")
    print()
    render_tab_B1()
    render_tab_B2_B3()
    render_tab_B4()
    render_tab_B5()


if __name__ == "__main__":
    main()
